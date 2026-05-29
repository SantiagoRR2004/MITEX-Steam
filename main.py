import dataAcquisition
import dataHandling
import topicModeling
import extractiveSummary
import topicModeling
import rag
import LLMManager
import decoders
import models
import random
import torch
import json
import os


def updateData(forceRefresh=False):
    """
    Update all the data needed for the RAG.

    Args:
        - forceRefresh (bool): Whether to force refresh the data. Default is False.
            Takes a long time if set to True.

    Returns:
        - None.
    """
    dataAcquisition.updateVideogames()
    dataAcquisition.getAllGames(forceRefresh=forceRefresh)
    print("Cleaning")
    dataHandling.cleanData(forceRefresh=forceRefresh)
    print("Topic modeling")
    topicModeling.completeTopicModelingPipeline(forceRefresh=forceRefresh)
    print("Extractive summarization")
    extractiveSummary.extractiveSummary(forceRefresh=forceRefresh)
    print("Adding documents to collection")
    rag.addDocsToCollection(forceRefresh=forceRefresh)


class Orchestrator:

    def main(self, q: str) -> str:
        """
        Execute the full consult from the user query to the final response.

        Args:
            - q (str): The original query from the user.
        Returns:
            - str: The final response to the user query.
        """
        currentDirectory = os.path.dirname(os.path.abspath(__file__))
        logFile = os.path.join(currentDirectory, "completeExecutions.json")
        self.completeExecution = {}

        responseJson = self.node1(q)

        if responseJson["Action"] == "rag":
            # Retrieve relevant documents from the collection
            results = list(rag.rrf(q, nResults=3))
            self.completeExecution["Retrieved Documents"] = results

            # Pass the retrieved documents to node 2
            finalOutput = self.node2(q, results)
        else:
            finalOutput = self.node2(q, [])

        # Save the complete execution log
        if os.path.exists(logFile):
            with open(logFile, "r", encoding="utf-8") as f:
                logs = json.load(f)
        else:
            logs = {}

        logs[q] = self.completeExecution

        with open(logFile, "w", encoding="utf-8") as f:
            json.dump(logs, f, indent=4, ensure_ascii=False)
            f.write("\n")

        return finalOutput

    def node1(self, userMessage: str) -> dict:
        """
        Node 1 uses the LLM to decide which action to take.

        Args:
            - userMessage (str): The message from the user that contains the query or task.

        Returns:
            - dict: A dictionary containing the thought process, the action to take.
        """
        systemPrompt = (
            "You are the brain of a multi-agent system. Your only function is to classify the user's query. You have two actions available:\n"
            " - 'rag': If the user's query is related to videogames and you think that retrieving relevant documents from the collection would help answer the query, choose this action.\n"
            " - 'nothing': If the user's query is not related to videogames or you think that retrieving documents would not help answer the query, choose this action. \n\n"
            "REQUIRED STRUCTURE in JSON format:\n"
            '{"Thinking": "[Explain here why you choose the tool.]", '
            '"Action": "[rag or nothing]" '
            "}\n"
        )

        self.completeExecution["Node 1 Prompt"] = systemPrompt

        messages = [
            {"role": "system", "content": systemPrompt},
            {"role": "user", "content": userMessage},
        ]

        sequence = decoders.TokenSequenceConstraint(
            wantedText=[
                '{"Thinking": "',
                decoders.EndingText('"', models.GEN_TOKENIZER),
                ', "Action": "',
                decoders.MultipleTokenOptions(
                    ["nothing", "rag"], tokenizer=models.GEN_TOKENIZER
                ),
                '"}',
            ],
            tokenizer=models.GEN_TOKENIZER,
        )

        # Use the model
        d = decoders.FormattedDecoder(models.GEN_TOKENIZER, sequence, jsonFormat=True)
        m = LLMManager.LLMManager(d)
        d.setManager(m)

        # Generate the response
        tokens = m.processPrompt(messages)

        while not d.finished:
            tokens = m.decode(torch.tensor([[tokens[-1]]]))

        finalText = m.decodeTokens(tokens)
        output = json.loads(finalText)

        self.completeExecution["Node 1 Output"] = output

        return output

    def node2(self, q: str, documents: list) -> str:
        """
        Node 2 is the final node that receives the original query
        and all the retrieved documents.

        Args:
            - q (str): The original query from the user.
            - documents (list): The list of retrieved documents.

        Returns:
            - str: The final response.
        """
        systemPrompt = (
            "You are a helpful assistant. Your task is to answer the user's query."
        )
        self.completeExecution["Node 2 Prompt"] = systemPrompt

        messages = [{"role": "system", "content": systemPrompt}]

        if len(documents) > 0:
            documentsText = "\n\n".join(
                [f"Document {i+1}:\n{doc}" for i, doc in enumerate(documents)]
            )
            self.completeExecution["Node 2 Retrieved Documents"] = documentsText
            systemPrompt += f"\n\nThe following documents are relevant to the user's query:\n\n{documentsText}"

        messages = [{"role": "system", "content": systemPrompt}]
        messages.append({"role": "user", "content": q})

        # Use the model
        m = LLMManager.LLMManager(decoders.SamplingDecoder())
        finalResponse = m.processPrompt(messages, maxTokens=1000)
        response = m.decodeTokens(finalResponse)

        self.completeExecution["Node 2 Output"] = response
        return response


if __name__ == "__main__":
    # updateData(False)

    currentDirectory = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(currentDirectory, "queries.json"), "r") as f:
        queries = json.load(f)

    for q in queries:
        print(q)
        print(Orchestrator().main(q))
        print("\n\n")
