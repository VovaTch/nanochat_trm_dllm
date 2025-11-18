from typing import Sequence
import torch


class LinearSampleScheduler:
    """
    Simple mask that performs random covering based on Bernoulli distribution, with
    p linear decreasing from 1 to 0.
    """

    def __init__(self, num_steps: int) -> None:
        """
        Initialize the scheduler with the given number of steps.

        Args:
            num_steps (int): The total number of steps for the scheduler.
        """
        super().__init__()
        self._num_steps = num_steps

    def get_mask(  # type: ignore
        self, step_idx: int, mask: Sequence[bool], logits: torch.Tensor | None = None
    ) -> torch.Tensor:
        prev_mask_torch = torch.tensor(mask).unsqueeze(0)
        if step_idx > self._num_steps:
            raise ValueError("Step must be less than the number of steps.")
        uncover_probability = step_idx / self._num_steps
        new_mask = torch.bernoulli(
            torch.ones_like(prev_mask_torch) * uncover_probability
        )
        return new_mask

    @property
    def steps(self) -> int:
        """
        Returns the number of steps in the sampling process.

        Returns:
            int: The number of steps.
        """
        return self._num_steps
