import dataAcquisition
import dataHandling
import topicModeling
import extractiveSummary

if __name__ == "__main__":
    forceRefresh = False

    dataAcquisition.updateVideogames()
    dataAcquisition.getAllGames(forceRefresh=forceRefresh)
    dataHandling.onlyEnglish(forceRefresh=forceRefresh)
    if forceRefresh:
        topicModeling.basicTopicModeling()
    extractiveSummary.extractiveSummary(forceRefresh=forceRefresh)
