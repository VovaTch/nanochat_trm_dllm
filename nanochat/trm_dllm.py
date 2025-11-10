"""
Tiny Recurrent Model (TRM) with a core of an LLM diffusion model. Will use the dataset targets, with thet appropriate
mask.
"""

from dataclasses import dataclass
from functools import partial
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.adamw import DistAdamW
from nanochat.common import get_dist_info
from nanochat.gpt import MLP, apply_rotary_emb, norm
from nanochat.muon import DistMuon, Muon


@dataclass
class TRDLMConfig:
    n_layer: int = 2
    n_embd: int = 512
    n_head: int = 8
    vocab_size: int = 50304
    sequence_len: int = 1024
    n_kv_head: int = 6
    y_loop: int = 3
    z_loop: int = 6
    sup_steps: int = 8


class FullSelfAttention(nn.Module):
    def __init__(self, config: TRDLMConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

    def forward(self, x: torch.Tensor, cos_sin: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()

        # Project the input to get queries, keys, and values
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Apply Rotary Embeddings to queries and keys to get relative positional encoding
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(
            k, cos, sin
        )  # QK rotary embedding
        q, k = norm(q), norm(k)  # QK norm
        q, k, v = (
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
        )  # make head be batch dim, i.e. (B, T, H, D) -> (B, H, T, D)

        y = F.scaled_dot_product_attention(q, k, v)
        # Re-assemble the heads side by side and project back to residual stream
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class Block(nn.Module):
    def __init__(self, config: TRDLMConfig, layer_idx: int) -> None:
        super().__init__()
        self.attn = FullSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor, cos_sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(norm(x), cos_sin)
        x = x + self.mlp(norm(x))
        return x


class DiffusionTransformerCore(nn.Module):
    def __init__(self, config: TRDLMConfig) -> None:
        super().__init__()
        self._config = config
        self._transformers = nn.ModuleList(
            [Block(config, layer_idx) for layer_idx in range(config.n_layer)]
        )

        self._y_init = nn.Buffer(torch.randn((1, 1, config.n_embd)), persistent=True)
        self._z_init = nn.Buffer(torch.randn((1, 1, config.n_embd)), persistent=True)
        self.rotary_seq_len = (
            config.sequence_len * 10
        )  # 10X over-compute should be enough, TODO make nicer?

    @property
    def y_init(self) -> nn.Buffer:
        return self._y_init

    @property
    def z_init(self) -> nn.Buffer:
        return self._z_init

    def init_weights(self) -> None:
        self.apply(self._init_weights)
        # zero out c_proj weights in all blocks
        for block in self._transformers:  # type: ignore
            torch.nn.init.zeros_(block.mlp.c_proj.weight)  # type: ignore
            torch.nn.init.zeros_(block.attn.c_proj.weight)  # type: ignore
        # init the rotary embeddings
        head_dim = self._config.n_embd // self._config.n_head
        cos, sin = self._precompute_rotary_embeddings(
            self.rotary_seq_len, head_dim, device="cpu"
        )
        self.cos, self.sin = cos, sin

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            # https://arxiv.org/pdf/2310.17813
            fan_out = module.weight.size(0)
            fan_in = module.weight.size(1)
            std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=1.0)

    def _precompute_rotary_embeddings(
        self, seq_len: int, head_dim: int, base: int = 10000, device: str | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if device is None:
            device = self._transformers[0].device.type  # type: ignore
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()  # keep them in bfloat16
        cos, sin = (
            cos[None, :, None, :],
            sin[None, :, None, :],
        )
        return cos, sin

    def forward(
        self, x: torch.Tensor | None, y: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        _, seq_len, _ = y.shape
        if x is None:
            if y.shape != z.shape:
                raise ValueError(
                    f"y and z must have the same shape, got y shape {y.shape} and z shape {z.shape}"
                )

            sum_in = y + z

        else:
            if x.shape != y.shape or y.shape != z.shape:
                raise ValueError(
                    f"x, y and z must have the same shape, got x shape {x.shape}, y shape {y.shape} "
                    f"and z shape {z.shape}"
                )

            sum_in = x + y + z

        cos_sin = (
            self.cos[:, 0 : 0 + seq_len],
            self.sin[:, 0 : 0 + seq_len],
        )  # truncate cache to current sequence length
        for block in self._transformers:
            sum_in = block(sum_in, cos_sin)
        sum_out = norm(sum_in)
        return sum_out


class InputEmbedding(nn.Module):
    def __init__(self, embedding_dim: int, vocab_size: int) -> None:
        super().__init__()
        self._embedding_dim = embedding_dim
        self._vocab_size = vocab_size
        self._embedding = nn.Embedding(vocab_size + 1, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._embedding(x)


class LinearQOutputHead(nn.Module):
    def __init__(self, hidden_dim: int, seq_length: int) -> None:
        super().__init__()
        self._hidden_dim = hidden_dim
        self._seq_length = seq_length

        layers = []
        layers.append(nn.Linear(hidden_dim, 1))
        layers.append(nn.Linear(seq_length, 1))

        self._layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._layers[0](x)
        x = x.view(x.shape[0], -1)
        return self._layers[1](x)


# TRDLM model; trying Karpathy's interface for plug and playing in this repo
class TRDLM(nn.Module):
    def __init__(self, config: TRDLMConfig) -> None:
        super().__init__()
        self._config = config
        self._core = DiffusionTransformerCore(config)
        self._input_embedding = InputEmbedding(config.n_embd, config.vocab_size)
        self._output_head = nn.Linear(config.n_embd, config.vocab_size)
        self._q_output_head = LinearQOutputHead(config.n_embd, config.sequence_len)

    def init_weights(self) -> None:
        self._core.init_weights()
        torch.nn.init.zeros_(self._output_head.weight)
        if self._input_embedding._embedding.weight.device.type == "cuda":
            self._input_embedding.to(dtype=torch.bfloat16)

    @property
    def y_init(self) -> torch.Tensor:
        return self._core.y_init

    @property
    def z_init(self) -> torch.Tensor:
        return self._core.z_init

    @property
    def config(self) -> TRDLMConfig:
        return self._config

    def get_device(self) -> str:
        return str(self._input_embedding._embedding.weight.device.type)

    def estimate_flops(self) -> int:
        nparams = sum(p.numel() for p in self.parameters())
        nparams_embedding = self._input_embedding._embedding.weight.numel()
        len, h, q, t = (
            self._config.n_layer,
            self._config.n_head,
            self._config.n_embd // self._config.n_head,
            self._config.sequence_len,
        )
        num_flops_per_token = (
            6 * (nparams - nparams_embedding)
            + 12 * len * h * q * t * self._config.y_loop * self._config.z_loop
        )
        return num_flops_per_token

    def setup_optimizers(
        self,
        unembedding_lr: float = 0.004,
        embedding_lr: float = 0.2,
        matrix_lr: float = 0.02,
        weight_decay: float = 0.0,
    ) -> tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
        model_dim = self._config.n_embd
        ddp, rank, _, _ = get_dist_info()
        # Separate out all parameters into 3 groups (matrix, embedding, lm_head)
        matrix_params = list(self._core._transformers.parameters())
        embedding_params = list(self._input_embedding.parameters())
        lm_head_params = list(self._output_head.parameters())
        q_output_head_params = list(self._q_output_head.parameters())
        assert len(list(self.parameters())) == len(matrix_params) + len(
            embedding_params
        ) + len(lm_head_params) + len(q_output_head_params)
        # Create the AdamW optimizer for the embedding and lm_head
        # Scale the LR for the AdamW parameters by ∝1/√dmodel (having tuned the LRs for 768 dim model)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        if rank == 0:
            print(
                f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}"
            )
        adam_groups = [
            dict(params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale),
            dict(params=embedding_params, lr=embedding_lr * dmodel_lr_scale),
            dict(params=q_output_head_params, lr=unembedding_lr * dmodel_lr_scale),
        ]
        adamw_kwargs = dict(betas=(0.8, 0.95), eps=1e-10, weight_decay=weight_decay)
        AdamWFactory = DistAdamW if ddp else partial(torch.optim.AdamW, fused=True)
        adamw_optimizer: torch.optim.Optimizer = AdamWFactory(adam_groups, **adamw_kwargs)  # type: ignore
        # Create the Muon optimizer for the linear layers
        muon_kwargs = dict(lr=matrix_lr, momentum=0.95)
        MuonFactory = DistMuon if ddp else Muon
        muon_optimizer: torch.optim.Optimizer = MuonFactory(matrix_params, **muon_kwargs)  # type: ignore
        # Combine them the two optimizers into one list
        optimizers = [adamw_optimizer, muon_optimizer]
        for opt in optimizers:
            for group in opt.param_groups:
                group["initial_lr"] = group["lr"]
        return tuple(optimizers)  # type: ignore

    def latent_recursion(
        self, input: torch.Tensor, output: torch.Tensor, latent: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for _ in range(self._config.z_loop):
            latent = self._core(input, output, latent)
        output = self._core(None, output, latent)
        return output, latent

    def deep_recursion(
        self, input: torch.Tensor, output: torch.Tensor, latent: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        input = self._input_embedding(input)
        with torch.no_grad():
            for _ in range(self._config.y_loop - 1):
                output, latent = self.latent_recursion(input, output, latent)
        # output = output.detach()
        # latent = latent.detach()
        output, latent = self.latent_recursion(input, output, latent)
        logits = self._output_head(output)
        q_stop = self._q_output_head(output)
        return (
            output.detach(),
            latent.detach(),
            logits,
            q_stop,
        )

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None = None,
        z: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if y is None:
            y = self._core.y_init.repeat((x.shape[0], x.shape[1], 1)).to(
                self.get_device()
            )
        if z is None:
            z = self._core.z_init.repeat((x.shape[0], x.shape[1], 1)).to(
                self.get_device()
            )

        _, _, output, q_stop = self.deep_recursion(x, y, z)
        return output, q_stop

    @staticmethod
    def get_loss(
        output: torch.Tensor,
        q_stop: torch.Tensor,
        mask: torch.Tensor,
        target: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:

        loss_cls = F.cross_entropy(
            output[~mask], target[~mask].long(), reduction=reduction
        )
        q_target_collector = []
        for i in range(output.shape[0]):

            q_stop_target_ind = torch.all(
                torch.argmax(output[i][~mask[i]], dim=1) == target[i][~mask[i]]
            )
            q_target_collector.append(q_stop_target_ind)

        q_stop_target = torch.stack(q_target_collector)
        loss_q = F.binary_cross_entropy_with_logits(
            q_stop.squeeze(),
            q_stop_target.float().squeeze(),
            reduction=reduction if reduction != "none" else "mean",
        )  # TODO: incorrect, but should work
        return loss_cls + loss_q
