from .abstractDecoder import AbstractDecoder
import torch
import json


class EndingText:
    def __init__(self, text: str, tokenizer) -> None:
        """
        Initialize the EndingText.

        Args:
            - text (str): The string that will signal the end of the generated text.
            - tokenizer: The tokenizer to use for encoding the text.

        Returns:
            - None
        """
        self.text = text
        self.tokenIds = tokenizer.encode(text, add_special_tokens=False)[0]
        assert isinstance(self.tokenIds, int), "EndingText should be a single token"


class MultipleTokenOptions:
    def __init__(self, options: list[str], tokenizer) -> None:
        """
        Initialize the MultipleTokenOptions.

        If an option is an smaller version of another, the second
        one will be ignored.

        Args:
            - options (list[str]): A list of possible token sequences (as strings).
            - tokenizer: The tokenizer to use for encoding the options.

        Returns:
            - None
        """
        for i in range(len(options)):
            options[i] = tokenizer.encode(options[i], add_special_tokens=False)

        self.options = options

    def nextPossibleTokens(self) -> list[int]:
        return list(set(option[0] for option in self.options))

    def chosenToken(self, tokenId: int) -> bool:
        """
        Handle the chosen token and update the options accordingly.

        Returning True means that this class should not be used to
        check the next token.

        Args:
            - tokenId (int): The ID of the chosen token.

        Returns:
            - bool: True if one of the options is fully matched, False otherwise.
        """
        newOptions = []
        for i in range(len(self.options)):
            if self.options[i][0] == tokenId:
                self.options[i].pop(0)
                newOptions.append(self.options[i])

        self.options = newOptions

        # If no options remain, raise error
        if self.options == []:
            raise ValueError(
                "No valid options remain for choosing tokenId: " + str(tokenId)
            )

        # If one of the options is empty, return True
        for option in self.options:
            if option == []:
                return True

        return False


class TokenSequenceConstraint:
    def __init__(self, wantedText: list[str], tokenizer) -> None:

        # Turn basic text into token ids
        for i in range(len(wantedText)):
            if isinstance(wantedText[i], str):
                wantedText[i] = tokenizer.encode(
                    wantedText[i], add_special_tokens=False
                )

        self.wantedText = wantedText

    def nextPossibleTokens(self) -> list[int]:
        nextItem = self.wantedText[0]

        if isinstance(nextItem, MultipleTokenOptions):
            return nextItem.nextPossibleTokens()

        else:
            return []

    def chosenToken(self, tokenId: int) -> bool:
        """
        Handle the chosen token and update the wantedText accordingly.

        Returning True means that the wantedText is fully matched and the
        decoder should stop checking the constraints.

        Args:
            - tokenId (int): The ID of the chosen token.

        Returns:
            - bool: True if the wantedText is fully matched, False otherwise.
        """
        nextItem = self.wantedText[0]

        if isinstance(nextItem, MultipleTokenOptions):
            # Handle multiple token options
            if nextItem.chosenToken(tokenId):
                self.wantedText.pop(0)

        return self.wantedText == []


class FormattedDecoder(AbstractDecoder):
    topP: float = 0.9

    def __init__(
        self,
        tokenizer,
        wantedSequence: TokenSequenceConstraint,
        jsonFormat: bool = False,
    ) -> None:
        """
        Initialize the FormattedDecoder.

        Args:
            - tokenizer: The tokenizer to use for decoding tokens.
            - wantedSequence: The token sequence constraint to follow.

        Returns:
            - None
        """
        self.tokenizer = tokenizer
        self.wantedSequence = wantedSequence
        self.finished = False
        self.jsonFormat = jsonFormat

        self.jsonDecoder = json.JSONDecoder()

    def setManager(self, manager) -> None:
        """
        Set the manager for the decoder.

        Args:
            - manager: The manager that will be using this decoder.

        Returns:
            - None
        """
        self.manager = manager

    def chooseNextToken(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Choose next token constrained by the wanted sequence.

        If a `wantedSequence` is set, only the allowed token ids returned by
        `wantedSequence.nexPossibleTokens()` are considered. The selection is
        done with the same top-p nucleus sampling used by the sampling
        decoder. If no constraint exists or `nexPossibleTokens()` returns an
        empty list, the full vocabulary is used.

        Args:
            - logits (torch.Tensor): Model logits for the current step (batch x vocab).

        Returns:
            - torch.Tensor: Chosen token id as a (1,1) tensor.
        """
        if self.finished:
            # Return end of sequence
            return torch.tensor([[self.tokenizer.eos_token_id]])

        nextWantedList = self.wantedSequence.wantedText
        nextWantedItem = nextWantedList[0]

        # If next is list make it decode the list
        # It is a list of token ids that we want to be decoded as they are, without sampling
        if isinstance(nextWantedItem, list):
            # wantedText is a list of token ids with shape (nTokens,)
            # We need a tensor with shape (batch_size, nTokens)
            tokens = torch.tensor([nextWantedItem])
            nextWantedList.pop(0)
            self.manager.generatedTokenIds.extend(tokens[0].tolist())

            if len(nextWantedList) == 0:
                self.finished = True

            # Make the cache be updated with the new tokens
            outputs = self.manager.MODEL(
                input_ids=tokens, use_cache=True, past_key_values=self.manager.kvCache
            )

            self.manager.kvCache = outputs.past_key_values

            return self.chooseNextToken(outputs.logits[:, -1, :])

        # Flatten logits to 1D (vocabulary dimension)
        logitsFlat = logits.view(-1)

        # Remove end of sequence token from allowed tokens to prevent early termination
        logitsFlat[self.tokenizer.eos_token_id] = float("-inf")

        # Get allowed token ids from the constraint
        allowed = self.wantedSequence.nextPossibleTokens()

        if len(allowed) == 0:
            # Empty list from nexPossibleTokens means "no specific constraint",
            # allow any token (this covers UnvalidToken behavior).
            filteredProbs = logitsFlat
            filteredIndices = torch.arange(logitsFlat.shape[-1])
        else:
            allowed_tensor = torch.tensor(
                allowed,
            )
            filteredProbs = logitsFlat[allowed_tensor]

            # Use argmax within the allowed tokens
            best_idx = torch.argmax(filteredProbs)
            filteredProbs = filteredProbs[best_idx : best_idx + 1]
            filteredIndices = allowed_tensor[best_idx : best_idx + 1]

        # Softmax
        filteredProbs = torch.softmax(filteredProbs, dim=-1)

        sortedProbs, sortedIndices = torch.sort(filteredProbs, descending=True)

        # Calculate cumulative probabilities
        cumulativeProbs = torch.cumsum(sortedProbs, dim=-1)
        cutoff = cumulativeProbs > self.topP
        cutoff[..., 1:] = cutoff[..., :-1].clone()
        cutoff[..., 0] = 0

        sortedProbs[cutoff] = 0
        sortedProbs = sortedProbs / sortedProbs.sum(dim=-1, keepdim=True)

        nextToken = torch.multinomial(sortedProbs, num_samples=1)
        tokenId = int(filteredIndices[sortedIndices.gather(-1, nextToken)].item())

        if isinstance(nextWantedItem, EndingText):
            # We check if the token has a quote
            decodedToken = self.tokenizer.decode([tokenId])

            # Check that we are currently on a broken string
            brokenString = True
            if self.jsonFormat:
                partialText = self.manager.decodeTokens(
                    self.manager.generatedTokenIds + [tokenId]
                )
                try:
                    self.jsonDecoder.raw_decode(partialText)
                except json.JSONDecodeError as e:
                    if "Unterminated string starting" not in e.msg:
                        brokenString = False

            if nextWantedItem.text in decodedToken or (
                self.jsonFormat and not brokenString
            ):

                # The string finishes the string with "
                nextWantedList.pop(0)
                if len(nextWantedList) == 0:
                    self.finished = True
                return torch.tensor([[nextWantedItem.tokenIds]])

        # Inform the constraint about the chosen token so it can advance
        self.finished = self.wantedSequence.chosenToken(tokenId)

        return torch.tensor([[tokenId]])
