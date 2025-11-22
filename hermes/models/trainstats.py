import pandas as pd
from pathlib import Path

class TrainStats:
    def __init__(self, envName:str, model:str):
        self.model = model
        self.envName = envName
        self.__initFolder()

    def __initFolder(self):
        statFolder = Path(f"environnements/{self.envName}/stats")
        statFolder.mkdir(exist_ok=True)
        self.folder = statFolder

    def removeTrainStat(self):
        fileName = f"environnements/{self.envName}/stats/{self.model}-train.csv"
        file = Path(fileName)
        if file.exists():
            file.unlink()


    def writeTrainStat(self, data:dict):
        df = pd.DataFrame(data, index=[0])

        fileName = f"environnements/{self.envName}/stats/{self.model}-train.csv"
        file = Path(fileName)
        if file.exists():
            dfOld = pd.read_csv(fileName)
            df = pd.concat([dfOld, df], ignore_index=True)

        df.to_csv(fileName, index=False)

    def writeClassificationReport(self, cr:dict):
        data = {
            'classe' : [],
            'precision' : [],
            'recall' : [],
            'f1-score' : []
        }

        for label in cr:
            if isinstance(cr[label], dict):
                data["classe"].append(label)
                data["precision"].append(cr[label]["precision"])
                data["recall"].append(cr[label]["recall"])
                data["f1-score"].append(cr[label]["f1-score"])

        df = pd.DataFrame(data, columns=["classe","precision","recall","f1-score"])
        df.to_csv(f"environnements/{self.envName}/stats/{self.model}-classificationreport.csv")

    def writeConfusionMatrix(self, cm):
        cm_df = pd.DataFrame(cm)
        cm_df.to_csv(f"environnements/{self.envName}/stats/{self.model}-confusionmatrix.csv", index=False)
    