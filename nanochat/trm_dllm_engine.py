from dataclasses import dataclass, field
import math
from typing import Any, Iterator, Protocol, Sequence

import torch
import torch.nn.functional as F

from nanochat.tokenizer import RustBPETokenizer
from nanochat.trm_dllm import TRDLM


class SampleScheduler(Protocol):
    def get_mask(
        self, step_idx: int, mask: Sequence[bool], logits: torch.Tensor | None = None
    ) -> list[bool]: ...

    @property
    def steps(sellf) -> int: ...


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
        scheduler: SampleScheduler,
        max_seq_len: int = 1024,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.scheduler = scheduler

    def _get_special_token(self, s: str) -> int:
        return self.tokenizer.encode_special(s)

    @torch.inference_mode()  # TODO: everything in the generate part is wrong
    def generate(
        self,
        tokens: Sequence[int],
        temperature: float = 1.0,
        top_k: int | None = None,
        seed: int = 42,
        **kwargs,
    ) -> Iterator[tuple[list[int], list[int]]]:
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
        mask = [True] * init_len + [False] * (max_seq_len - init_len)

        current_sequence = list(tokens) + [vocab_size] * (len(tokens) - max_seq_len)
        current_sequence = torch.tensor(current_sequence).to(device)
        current_sequence = (
            torch.cat(
                [
                    current_sequence,
                    torch.ones(max_seq_len - len(current_sequence)).to(device)
                    * vocab_size,
                ]
            )
            .to(dtype=torch.long)
            .unsqueeze(0)
        )

        y = self.model.y_init.repeat((1, max_seq_len, 1)).to(device)
        z = self.model.z_init.repeat((1, max_seq_len, 1)).to(device)

        row_states = [
            SeqState(
                current_tokens=current_sequence.tolist(),
                current_mask=mask,
                current_step=0,
                in_python_block=False,
                completed=False,
            )
        ]

        for step_idx in range(self.scheduler.steps):
            if step_idx > 0 and step_idx < self.scheduler.steps - 1:
                mask = (
                    self.scheduler.get_mask(step_idx, mask, current_sequence)
                    .to(dtype=torch.bool)
                    .squeeze()
                )
                mask = [True] * init_len + mask[init_len:max_seq_len].tolist()
            elif step_idx == self.scheduler.steps - 1:
                mask = [True] * max_seq_len
            mask_tensor = torch.tensor(mask).unsqueeze(0).to(device)
            current_sequence[~mask_tensor] = vocab_size
            y, z, output_logits, q_stop = self.model.deep_recursion(
                current_sequence, y, z
            )

            current_sequence = sample_sequence(
                current_sequence,
                output_logits,
                mask=mask_tensor,
                rng=rng,
                temperature=temperature,
                top_k=top_k,
            )

            row_states.append(
                SeqState(
                    current_tokens=current_sequence[0].tolist(),
                    current_mask=mask,
                    current_step=step_idx + 1,
                    completed=True if step_idx == self.scheduler.steps - 1 else False,
                )
            )

            yield current_sequence[0].tolist(), torch.tensor(mask).to(
                dtype=torch.int
            ).tolist()

            if torch.all(q_stop > 0):
                break

            if (~mask_tensor).sum() == 0:
                break

    def generate_batch(
        self, tokens, num_samples=1, **kwargs
    ) -> tuple[list[list[int]], list[Any]]:
        """
        Non-streaming batch generation that just returns the final token sequences.
        Returns a list of token sequences (list of lists of ints).
        Terminal tokens (assistant_end, bos) are not included in the results.
        """
        assistant_end = self.tokenizer.encode_special("<|assistant_end|>")
        bos = self.tokenizer.get_bos_token_id()
        results = [tokens.copy() for _ in range(num_samples)]
        masks = [[0] * len(tokens) for _ in range(num_samples)]
        completed = [False] * num_samples

        for sample_idx in range(num_samples):
            for gen_tokens, gen_mask in self.generate(tokens, **kwargs):
                if completed[sample_idx]:
                    break
                # Check for terminal tokens
                if (
                    assistant_end in gen_tokens[len(tokens) :]
                    or bos in gen_tokens[len(tokens) :]
                ):
                    if assistant_end in gen_tokens[len(tokens) :]:
                        term_idx = gen_tokens.index(assistant_end)
                    else:
                        term_idx = gen_tokens.index(bos)
                    results[sample_idx] = gen_tokens[:term_idx]
                    masks[sample_idx] = gen_mask[:term_idx]
                    completed[sample_idx] = True
                else:
                    results[sample_idx] = gen_tokens
                    masks[sample_idx] = gen_mask

        return results, masks
