from dotenv import load_dotenv
import decoders
import models
import torch
import time

load_dotenv(".env.local")


class LLMManager:

    TOKENIZER = models.GEN_TOKENIZER
    MODEL = models.GEN_MODEL

    # Check if the tokenizer has a specific end-of-turn token
    possibleEnd = TOKENIZER.convert_tokens_to_ids("<end_of_turn>")
    if possibleEnd is not None and possibleEnd != TOKENIZER.unk_token_id:
        eotTokenId = possibleEnd
    else:
        eotTokenId = -1
    del possibleEnd

    def __init__(self, decoder: decoders.AbstractDecoder) -> None:
        self.decoder = decoder
        self.kvCache = None
        self.ttft = 0
        self.inferenceSpeed = []
        self.allTokens: list[int] = []

    def processPrompt(
        self,
        messages: list[dict],
        maxTokens: int = 100,
    ) -> list[int]:
        """
        Process the input prompt and generate a response using the language model.

        Args:
            - messages (list[dict]): The input messages to be processed by the model.
            - maxTokens (int): The maximum number of tokens to generate in the response.
        Returns:
            - list[int]: A list of generated token IDs.
        """
        startTime = time.monotonic()
        # add_generation_prompt=True add </s><|assistant|>\n at the end of the prompt
        input = self.TOKENIZER.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )
        inputIds = input["input_ids"]

        self.generatedTokenIds = []

        if self.kvCache is None:
            inputIds = self.prefill(inputIds)
            self.generatedTokenIds.append(inputIds.item())
            self.ttft = time.monotonic() - startTime
            maxTokens -= 1

        # Generate the rest
        self.decode(inputIds, maxTokens)

        return self.generatedTokenIds

    def prefill(self, inputIds: torch.Tensor) -> int:
        """
        Prefill the model's key-value cache with the given input token IDs.

        Args:
            - inputIds (torch.Tensor): The token IDs to prefill the model's cache.

        Returns:
            - int: The ID of the next token.
        """
        self.allTokens.extend(inputIds[0].tolist())
        with torch.no_grad():
            outputs = self.MODEL(input_ids=inputIds)
            self.kvCache = outputs.past_key_values
            nextTokenId = self.decoder.chooseNextToken(outputs.logits[:, -1, :])

        return nextTokenId

    def decode(self, inputIds: list[int], maxTokens: int = 100) -> list[int]:
        """
        Generate a response by decoding the input token IDs using the model.

        Args:
            - inputIds (list[int]): The initial token IDs to start decoding from.
            - maxTokens (int): The maximum number of tokens to generate in the response.

        Returns:
            - list[int]: A list of token IDs representing the generated response.
        """
        # Check that generatedTokenIds is initialized
        if not hasattr(self, "generatedTokenIds"):
            self.generatedTokenIds = []

        self.allTokens.extend(inputIds[0].tolist())

        for _ in range(maxTokens):
            startTime = time.monotonic()

            with torch.no_grad():
                outputs = self.MODEL(
                    input_ids=inputIds, use_cache=True, past_key_values=self.kvCache
                )

            self.kvCache = outputs.past_key_values
            nextTokenId = self.decoder.chooseNextToken(outputs.logits[:, -1, :])

            self.inferenceSpeed.append(time.monotonic() - startTime)

            # Only use the last token for the next input
            inputIds = nextTokenId

            self.generatedTokenIds.append(nextTokenId.item())
            self.allTokens.append(nextTokenId.item())
            if nextTokenId.item() in [self.TOKENIZER.eos_token_id, self.eotTokenId]:
                break

        return self.generatedTokenIds

    def inyectToCache(self, text: str) -> None:
        """
        Inyect a text into the model's key-value cache.

        Args:
            - text (str): The text to be inyected into the cache.

        Returns:
            - None.
        """
        tokens = self.TOKENIZER(text, return_tensors="pt").input_ids
        self.allTokens.extend(tokens[0].tolist())
        with torch.no_grad():
            outputs = self.MODEL(
                input_ids=tokens, use_cache=True, past_key_values=self.kvCache
            )
            self.kvCache = outputs.past_key_values

    def resetCache(self) -> None:
        """
        Reset the model's key-value cache to its initial state.

        Args:
            - None.

        Returns:
            - None.
        """
        self.kvCache = None
        self.allTokens = []

    def decodeTokens(self, tokenIds: list[int]) -> str:
        """
        Decode a list of token IDs into a human-readable string.

        Args:
            - tokenIds (list[int]): A list of token IDs to be decoded.

        Returns:
            - str: The decoded string corresponding to the input token IDs.
        """
        return self.TOKENIZER.decode(tokenIds, skip_special_tokens=True)
