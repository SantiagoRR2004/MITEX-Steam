from abc import ABC, abstractmethod
import torch


class AbstractDecoder(ABC):
    @abstractmethod
    def chooseNextToken(self, logits: torch.Tensor) -> torch.Tensor:
        pass
