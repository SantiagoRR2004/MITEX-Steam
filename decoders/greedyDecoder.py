from .abstractDecoder import AbstractDecoder
import torch


class GreedyDecoder(AbstractDecoder):
    def chooseNextToken(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Choose the next token by selecting the one with the highest logit value.

        Args:
            - logits (torch.Tensor): The output logits from the model for the current step.

        Returns:
            - torch.Tensor: The token ID of the chosen next token.
        """
        return torch.argmax(logits, dim=-1, keepdim=True)
