import dataAcquisition
import dataHandling
import topicModeling
import extractiveSummary

if __name__ == "__main__":
    dataAcquisition.updateVideogames()
    dataAcquisition.getAllGames()
    dataHandling.onlyEnglish()
    topicModeling.basicTopicModeling()
    extractiveSummary.extractiveSummary()
