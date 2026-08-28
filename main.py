from TopSellers import dataAcquisition
from datetime import datetime
from bs4 import BeautifulSoup
import dataHandling
import topicModeling
import extractiveSummary
import rag
import LLMManager
import decoders
import requests
import difflib
import models
import pickle
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
    useSynthetic = True

    systemPrompt1 = (
        "You are the brain of a multi-agent system. Your only function is to classify the user's query.\n\n"
        "To help you decide, you should know that a 'Videogame Document' in our system contains the following structured information:\n"
        " - Title and Genres\n"
        " - Full Description of the game\n"
        " - Summary of Positive Reviews (bullet points)\n"
        " - Summary of Negative Reviews (bullet points)\n"
        " - Associated Reviews Topics/Themes with percentage of relevance (e.g., Theme: Sci-Fi, Description: ..., 85%)\n\n"
        "You have three actions available:\n"
        " - 'rag': If the query is broad, compares multiple games, or asks about genres/tropes, and you think retrieving parts of various documents will help answer it (e.g., 'What RPGs have good sci-fi stories?').\n"
        " - 'search': If the user explicitly asks for information, reviews, opinions, or themes of ONE specific videogame, and you can identify its name (e.g., 'What do people think about Cyberpunk 2077?'). You can ONLY search by the game name.\n"
        " - 'nothing': If the query is not related to videogames, is conversational, or doesn't require any document data (e.g., 'Hello', 'Who are you?').\n\n"
        "REQUIRED STRUCTURE in JSON format:\n"
        '{"Thinking": "[Explain here why you choose the tool based on the query and the document structure.]", '
        '"Action": "[rag or nothing or search]" '
        "}\n"
    )

    systemPrompt2 = (
        "You are an expert videogame assistant. Your task is to answer the user's query using the provided documents.\n\n"
        "ABOUT THE DOCUMENTS:\n"
        " - Each document contains: Title, Genres, Description, Positive/Negative Reviews, and Associated Topics.\n"
        " - NOTE: The 'Associated Topics' are extracted directly from user reviews, representing the key themes discussed by the community.\n"
        " - CRITICAL: You will often receive up to 3 documents, but the user's query might only target one or two specific game. "
        "Identify which document matches the user's intent, focus entirely on it, and IGNORE the other irrelevant documents. Do not try to combine them if it is not relevant.\n\n"
        "CRITICAL RULES:\n"
        "1. FOCUS & RELEVANCE: Answer using only the document(s) that actually matter for the query. If a document is irrelevant, disregard it completely.\n"
        "2. OBJECTIVITY: Rely strictly on the provided data. If the documents do not contain the answer, politely state that you lack enough information.\n"
        "3. NO CITATIONS: Integrate the facts naturally. Do NOT say 'According to Document 1...' or 'In the first game...'. Just talk about the game naturally.\n"
        "4. TONE: Concise, direct, and objective."
        "5. SUMMARIZATION: Do not give the document or a part of the document exactly as it is. Always rephrase and summarize the information in your own words. Do not use bullet points, just write normal paragraphs.\n"
    )

    systemPrompt4 = (
        "You are a helpful assistant. Your task is to identify the name of the videogame that the previous node is thinking is needed. If you are not sure about the name of the game, try to give your best guess.\n\n"
        "REQUIRED STRUCTURE in JSON format:\n"
        '{"Game": "[Name of the game]"}\n'
    )

    systemPromptS = (
        "You are simulating a HUMAN USER (a gamer) chatting with an AI videogame assistant. "
        "Your job is to keep the conversation going naturally based on the history.\n\n"
        "RULES FOR THE SIMULATION:\n"
        "1. ROLE: You are the one asking questions, looking for recommendations, or asking for opinions about games. "
        "NEVER answer your own questions, NEVER recommend games to the AI, and NEVER act like the assistant.\n"
        "2. STYLE: Write like a normal human in a chat. Keep it casual, relatively short (1-2 sentences), and direct.\n"
        "3. ACTIONS AVAILABLE:\n"
        "   - Ask a follow-up question about the game the assistant just mentioned (e.g., 'Does it have multiplayer?', 'Is the story good?').\n"
        "   - Ask for a recommendation based on a genre or vibe (e.g., 'Can you recommend a good indie horror game?').\n"
        "   - If the conversation feels naturally finished or you have no more questions, just type exactly ':q' to exit. Do not say goodbye or thank you, just ':q'.\n"
    )

    def createCaches(self) -> None:
        """
        Create the necessary caches for the nodes.

        Args:
            - None

        Returns:
            - None
        """
        cacheFolder = os.path.join(self.currentDirectory, "cache")
        os.makedirs(cacheFolder, exist_ok=True)

        self.caches = {
            "node1": None,
            "node2": None,
            "node4": None,
            "nodeS": None,
        }

        for nodeName in self.caches.keys():
            cacheFile = os.path.join(cacheFolder, f"{nodeName}.pkl")

            if os.path.exists(cacheFile):
                with open(cacheFile, "rb") as f:
                    self.caches[nodeName] = pickle.load(f)
            else:

                message = [
                    {
                        "role": "system",
                        "content": getattr(self, f"systemPrompt{nodeName[-1]}"),
                    }
                ]
                input = models.GEN_TOKENIZER.apply_chat_template(
                    message, add_generation_prompt=True, return_tensors="pt"
                )["input_ids"]
                m = LLMManager.LLMManager(decoders.SamplingDecoder())
                m.prefill(input)

                self.caches[nodeName] = m.kvCache

                with open(cacheFile, "wb") as f:
                    pickle.dump(self.caches[nodeName], f)

    def main(self, q: str) -> str:
        """
        Execute the full consult from the user query to the final response.

        Args:
            - q (str): The original query from the user.
        Returns:
            - str: The final response to the user query.
        """
        self.createCaches()

        logFile = os.path.join(self.currentDirectory, "completeExecutions.json")
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
            print()

            if self.useSynthetic:
                q = self.nodeSynth()
                self.completeExecution[f"{datetime.now()} Synthetic Query"] = q
                print(f"Simulated user response: {q}\n")
            else:
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
        self.completeExecution[f"{datetime.now()} Node 1 Prompt"] = self.systemPrompt1

        messages = copy.deepcopy(self.completeConversation)
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
        m.kvCache = copy.deepcopy(self.caches["node1"])
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
        self.completeExecution[f"{datetime.now()} Node 2 Prompt"] = self.systemPrompt2

        messages = copy.deepcopy(self.completeConversation)

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
        self.completeConversation.append({"role": "user", "content": q})

        # Use the model
        m = LLMManager.LLMManager(decoders.SamplingDecoder())
        m.kvCache = copy.deepcopy(self.caches["node2"])
        finalResponse = m.processPrompt(messages, maxTokens=1000)
        response = m.decodeTokens(finalResponse)

        # Update the main caché
        self.caches["node2"] = m.kvCache
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
        self.completeExecution[f"{datetime.now()} Node 4 Prompt"] = self.systemPrompt4
        messages = copy.deepcopy(self.completeConversation)
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
        m.kvCache = copy.deepcopy(self.caches["node4"])
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
        videogamesFile = os.path.join(
            self.currentDirectory, "TopSellers", "videogames.json"
        )
        with open(videogamesFile, "r", encoding="utf-8") as f:
            videogames = json.load(f)

        match = difflib.get_close_matches(
            videogameName, videogames.keys(), n=1, cutoff=0.6
        )  # 60% similarity threshold
        self.completeExecution[f"{datetime.now()} Node 5 Closest Match"] = match

        if match:
            documents = [rag.generateDocument(match[0])] if match else []
            self.completeExecution[f"{datetime.now()} Node 5 Retrieved Document"] = (
                documents[0] if documents else None
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

                videogameID = int(result["href"].split("/")[4])
                videogameFoundName = (
                    result.find("span", class_="title").text.replace("/", "").strip()
                )

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
                documents[0] if documents else None
            )

        return documents

    def nodeSynth(self) -> str:
        """
        This is a synthetic node that simulates a possible user response based on the conversation so far.

        Args:
            - None

        Returns:
            - str: The simulated user response.
        """
        messages = copy.deepcopy(self.completeConversation)

        m = LLMManager.LLMManager(decoders.SamplingDecoder())
        m.kvCache = copy.deepcopy(self.caches["nodeS"])
        finalResponse = m.processPrompt(messages, maxTokens=1000)
        response = m.decodeTokens(finalResponse)

        return response


if __name__ == "__main__":
    # updateData(False)

    currentDirectory = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(currentDirectory, "queries.json"), "r") as f:
        queries = json.load(f)

    for q in queries:
        print(q)
        Orchestrator().main(q)
        print("\n\n")
