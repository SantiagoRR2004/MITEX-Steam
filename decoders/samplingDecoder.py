from .abstractDecoder import AbstractDecoder
import torch


class SamplingDecoder(AbstractDecoder):
    topP: float = 0.9

    def __init__(self, temperature: float = 1.0) -> None:
        self.temperature = temperature

    def chooseNextToken(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Choose the next token by sampling from the probability distribution
        defined by the logits, applying top-p (nucleus) filtering.

        Args:
            - logits (torch.Tensor): The output logits from the model for the current step.

        Returns:
            - torch.Tensor: The token ID of the chosen next token.
        """
        probabilities = torch.softmax(logits / self.temperature, dim=-1)
        sortedProbs, sortedIndices = torch.sort(probabilities, descending=True)

        # Calculate cumulative probabilities
        cumulativeProbs = torch.cumsum(sortedProbs, dim=-1)

        cutoff = cumulativeProbs > self.topP
        # Shift the cutoff mask to the right to include the first token that exceeds topP
        cutoff[..., 1:] = cutoff[..., :-1].clone()
        cutoff[..., 0] = 0

        sortedProbs[cutoff] = 0
        sortedProbs = sortedProbs / sortedProbs.sum(dim=-1, keepdim=True)

        nextToken = torch.multinomial(sortedProbs, num_samples=1)
        return sortedIndices.gather(-1, nextToken)
