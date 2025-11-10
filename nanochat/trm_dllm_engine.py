from dataclasses import dataclass, field
import math
from typing import Sequence

import torch
import torch.nn.functional as F

from nanochat.tokenizer import RustBPETokenizer
from nanochat.trm_dllm import TRDLM


@torch.inference_mode()
def sample_sequence(
    prev_sequence: torch.Tensor,
    logits: torch.Tensor,
    mask: torch.Tensor | None = None,
    rng: torch.Generator | None = None,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> torch.Tensor:
    """
    Perform sequence sampling for a diffusion model

    Args:
        prev_sequence (torch.Tensor): The previous sequence
        logits (torch.Tensor): The logits from the model
        mask (torch.Tensor | None, optional): The mask to apply to the sequence. Defaults to None.
        rng (torch.Generator | None, optional): The random number generator to use. Defaults to None.
        temperature (float, optional): The temperature to use for sampling. Defaults to 1.0.
        top_k (int | None, optional): The top k to use for sampling. Defaults to None.

    Returns:
        torch.Tensor: The sampled sequence
    """
    if math.isclose(temperature, 0.0):

        sampled_sequence = torch.argmax(logits, dim=-1)
        if mask is None:
            return sampled_sequence
        else:
            seq = prev_sequence.clone()
            seq[~mask] = sampled_sequence[~mask]
            return seq

    if top_k is not None:
        k = min(top_k, logits.size(-1))
        vals, idx = torch.topk(logits, k, dim=-1)
        mask = torch.full_like(logits, float("-inf"))
        filtered_logits = mask.scatter_(1, idx, vals)
        probs = F.softmax(filtered_logits / temperature, dim=-1)
        sampled_sequence = torch.multinomial(probs, num_samples=1, generator=rng)

    else:
        vals = logits / temperature
        probs = F.softmax(vals, dim=-1)
        sampled_sequence = torch.multinomial(probs, num_samples=1, generator=rng)

    if mask is None:
        return sampled_sequence
    else:
        seq = prev_sequence.clone()
        seq[~mask] = sampled_sequence[~mask]
        return seq


@dataclass
class SeqState:
    current_tokens: list[int] = field(default_factory=list)
    current_mask: list[bool] = field(default_factory=list)
    current_step: int = 0
    in_python_block: bool = False
    python_expr_tokens: list[int] = field(default_factory=list)
    completed: bool = False


class TrdlmEngine:
    def __init__(
        self,
        model: TRDLM,
        tokenizer: RustBPETokenizer,
        scheduler,
        max_seq_len: int = 1024,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.scheduler = scheduler

    def _get_special_token(self, s: str) -> int:
        return self.tokenizer.encode_special(s)

    @torch.inference_mode()
    def generate(
        self,
        tokens: Sequence[int],
        temperature: float = 1.0,
        top_k: int | None = None,
        seed: int = 42,
    ) -> torch.Tensor:
        device = self.model.get_device()
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)

        python_start = self._get_special_token("<|python_start|>")
        python_end = self._get_special_token("<|python_end|>")
        output_start = self._get_special_token("<|output_start|>")
        output_end = self._get_special_token("<|output_end|>")
        assistant_end = self._get_special_token(
            "<|assistant_end|>"
        )  # if sampled, ends row
        bos = self.tokenizer.get_bos_token_id()  # if sampled, ends row

        max_seq_len = self.model.config.sequence_len
        init_len = len(tokens)
        vocab_size = self.model.config.vocab_size
        mask = [False] * max_seq_len

        current_sequence = list(tokens) + [vocab_size] * (len(tokens) - max_seq_len)

        y = self.model.y_init.repeat((1, max_seq_len, 1)).to(device)
        z = self.model.z_init.repeat((1, max_seq_len, 1)).to(device)

        for step_idx in range(self.scheduler.steps):
            mask = self.scheduler.get_mask(step_idx, mask)
            current_sequence[~mask] = vocab_size
            y, z, output_logits, q_stop = self.model.deep_recursion(
                torch.tensor(current_sequence).to(device), y, z
            )
