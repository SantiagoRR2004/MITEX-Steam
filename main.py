from datetime import datetime
from bs4 import BeautifulSoup
import dataAcquisition
import dataHandling
import topicModeling
import extractiveSummary
import rag
import LLMManager
import decoders
import requests
import difflib
import models
import torch
import copy
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

    currentDirectory = os.path.dirname(os.path.abspath(__file__))

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
        self.completeExecution = {f"{datetime.now()} Query": q}
        self.completeConversation = []

        # Load previous logs
        if os.path.exists(logFile):
            with open(logFile, "r", encoding="utf-8") as f:
                logs = json.load(f)
        else:
            logs = {}
        logs[str(datetime.now())] = self.completeExecution

        while q != ":q":

            responseJson = self.node1(q)
            documents = []

            if responseJson["Action"] == "rag":
                # Retrieve relevant documents from the collection
                documents = self.node3(q)

            elif responseJson["Action"] == "search":
                documents = self.node4(q)

            elif responseJson["Action"] == "nothing":
                pass

            else:
                raise ValueError(f"Invalid action: {responseJson['Action']}")

            finalOutput = self.node2(q, documents)

            with open(logFile, "w", encoding="utf-8") as f:
                json.dump(logs, f, indent=4, ensure_ascii=False)
                f.write("\n")

            print(finalOutput)
            q = input("Enter a new query (or ':q' to quit): ")

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
            " - 'search': If the user's query is related to videogames and you think that retrieving the specific document by stating the real name of the game, choose this action. \n\n"
            "REQUIRED STRUCTURE in JSON format:\n"
            '{"Thinking": "[Explain here why you choose the tool.]", '
            '"Action": "[rag or nothing or search]" '
            "}\n"
        )

        self.completeExecution[f"{datetime.now()} Node 1 Prompt"] = systemPrompt

        messages = copy.deepcopy(self.completeConversation)
        messages.insert(0, {"role": "system", "content": systemPrompt})
        messages.append({"role": "user", "content": userMessage})

        sequence = decoders.TokenSequenceConstraint(
            wantedText=[
                '{"Thinking": "',
                decoders.EndingText('"', models.GEN_TOKENIZER),
                ', "Action": "',
                decoders.MultipleTokenOptions(
                    ["nothing", "rag", "search"], tokenizer=models.GEN_TOKENIZER
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

        self.completeExecution[f"{datetime.now()} Node 1 Output"] = output

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
        self.completeExecution[f"{datetime.now()} Node 2 Prompt"] = systemPrompt

        if getattr(self, "mainCache", None) is None:
            messages = [{"role": "system", "content": systemPrompt}]
            self.mainCache = None
            self.completeConversation = []
        else:
            messages = []

        if len(documents) > 0:
            documentsText = "\n\n".join(
                [f"Document {i+1}:\n{doc}" for i, doc in enumerate(documents)]
            )
            self.completeExecution[f"{datetime.now()} Node 2 Retrieved Documents"] = (
                documentsText
            )
            finalText = f"The following documents could relevant to the user's query:\n\n{documentsText}\n\nUser query:\n\n"
        else:
            finalText = ""

        messages.append({"role": "user", "content": finalText + q})
        self.completeConversation.append({"role": "user", "content": finalText + q})

        # Use the model
        m = LLMManager.LLMManager(decoders.SamplingDecoder())
        m.kvCache = self.mainCache
        finalResponse = m.processPrompt(messages, maxTokens=1000)
        response = m.decodeTokens(finalResponse)

        # Update the main caché
        self.mainCache = m.kvCache
        self.completeConversation.append({"role": "assistant", "content": response})

        self.completeExecution[f"{datetime.now()} Node 2 Output"] = response
        return response

    def node3(self, q: str) -> list:
        """
        This is the RAG node that uses the query to
        retrieve the relevant documents.

        Args:
            - q (str): The original query from the user.

        Returns:
            - list: The list of retrieved documents.
        """
        results = list(rag.rrf(q, nResults=3))
        self.completeExecution[f"{datetime.now()} Retrieved Documents"] = results
        return results

    def node4(self, q: str) -> list:
        """
        This is the search node that asks the model for the name of the game
        and then retrieves the relevant document.

        Args:
            - q (str): The original query from the user.

        Returns:
            - list: The list with the retrieved document.
        """
        systemPrompt = (
            "You are a helpful assistant. Your task is to identify the name of the videogame that the previous node is thinking is needed. If you are not sure about the name of the game, try to give your best guess.\n\n"
            "REQUIRED STRUCTURE in JSON format:\n"
            '{"Game": "[Name of the game]"}\n'
        )

        self.completeExecution[f"{datetime.now()} Node 4 Prompt"] = systemPrompt

        messages = copy.deepcopy(self.completeConversation)

        messages.insert(0, {"role": "system", "content": systemPrompt})
        messages.append({"role": "user", "content": q})

        sequence = decoders.TokenSequenceConstraint(
            wantedText=[
                '{"Game": "',
                decoders.EndingText('"', models.GEN_TOKENIZER),
                "}",
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

        self.completeExecution[f"{datetime.now()} Node 4 Output"] = output

        return self.node5(output["Game"])

    def node5(self, videogameName: str) -> list:
        """
        This node receives the name of the videogame and retrieves the document.

        If the there is no close match for the name of the videogame,
        it performs a web search to find the correct name and then retrieves the document.

        Args:
            - videogameName (str): The name of the videogame.

        Returns:
            - list: The list with the retrieved document.
        """
        # Try to find the closest match
        videogamesFile = os.path.join(self.currentDirectory, "videogames.json")
        with open(videogamesFile, "r", encoding="utf-8") as f:
            videogames = json.load(f)

        match = difflib.get_close_matches(
            videogameName, videogames.keys(), n=1, cutoff=0.6
        )  # 60% similarity threshold
        self.completeExecution[f"{datetime.now()} Node 5 Closest Match"] = match

        if match:
            documents = [rag.generateDocument(match[0])] if match else []
            self.completeExecution[f"{datetime.now()} Node 5 Retrieved Document"] = (
                documents[0]
            )

        else:
            # Use web search
            params = {
                "term": videogameName,
                "f": "games",
                "cc": "US",
                "l": "english",
                "hidef2p": 1,
            }

            response = requests.get(
                "https://store.steampowered.com/search/results/", params=params
            )

            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

            # Div with id="search_result_container"
            resultsContainer = soup.find("div", id="search_result_container")

            newVideogames = {}

            # Iterate through a class="search_result_row ds_collapse_flag"
            for result in resultsContainer.find_all(
                "a", class_="search_result_row ds_collapse_flag"
            ):

                videogameID = int(result["data-ds-appid"])
                videogameFoundName = result.find("span", class_="title").text.strip()

                if videogameFoundName not in videogames:
                    newVideogames[videogameFoundName] = videogameID

            self.completeExecution[f"{datetime.now()} Node 5 New Videogames Found"] = (
                newVideogames
            )

            # Update videogames.json
            videogames.update(newVideogames)
            videogames = dict(sorted(videogames.items(), key=lambda x: x[0].lower()))
            with open(videogamesFile, "w", encoding="utf-8") as f:
                json.dump(videogames, f, indent=2, ensure_ascii=False)
                f.write("\n")

            # Generate documents
            updateData(forceRefresh=False)

            match = difflib.get_close_matches(
                videogameName, newVideogames.keys(), n=1, cutoff=0.6
            )  # 60% similarity threshold
            self.completeExecution[
                f"{datetime.now()} Node 5 Closest Match After Search"
            ] = match

            documents = [rag.generateDocument(match[0])] if match else []
            self.completeExecution[f"{datetime.now()} Node 5 Retrieved Document"] = (
                documents[0]
            )

        return documents


if __name__ == "__main__":
    # updateData(False)

    currentDirectory = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(currentDirectory, "queries.json"), "r") as f:
        queries = json.load(f)

    for q in queries:
        print(q)
        Orchestrator().main(q)
        print("\n\n")
