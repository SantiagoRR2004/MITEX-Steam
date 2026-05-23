import dataAcquisition
import dataHandling
import topicModeling
import extractiveSummary
import LLMManager
import decoders


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
    dataHandling.onlyEnglish(forceRefresh=forceRefresh)
    if forceRefresh:
        topicModeling.basicTopicModeling()
    extractiveSummary.extractiveSummary(forceRefresh=forceRefresh)


class Orchestrator:
    def __init__(self) -> None:
        self.manager = LLMManager.LLMManager(decoders.GreedyDecoder())

    def main(self, q: str) -> str:
        """
        Execute the full consult from the user query to the final response.

        Args:
            - q (str): The original query from the user.
        Returns:
            - str: The final response to the user query.
        """
        pass


if __name__ == "__main__":
    updateData()
