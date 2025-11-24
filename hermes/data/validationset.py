import pandas as pd
import os
from hermes.core import Hermes
from hermes.models.classifier import Classifier
import json 

class ValidationSet:
    def __init__(self, envName:str, score:bool=False):
        self.envName = envName
        self.fileName = fileName = f"environnements/{self.envName}/validationset.csv"
        self.df = None
        self.__loadFile(score=score)
        

    def __loadFile(self, score:bool=False):
        if os.path.isfile(self.fileName):
            self.df = pd.read_csv(self.fileName)
            self.df['id'] = range(len(self.df))
            if score:
                #Réalisation de prédictions
                hermes = Hermes(envName=self.envName)

                self.df['classNamePredicted'] = "NC"
                self.df["labelsPredicted"] = "NC"
                self.df["score"] = 0


                for index, row in self.df.iterrows():
                    score = 0
                    r = hermes.analyseCommand(command=row["command"])
                    if row["className"] == r["action"]:
                        score = score+0.5

                    self.df.loc[index, "classNamePredicted"] = r["action"]

                    trueLabels = json.loads(row["labels"])

                    labels = {}
                    for argName in r["arguments"]:
                        correct = 0
                        if argName in trueLabels and trueLabels[argName] == r["arguments"][argName]:
                            correct = 1

                        labels[argName] = {
                            "value" : r["arguments"][argName],
                            "correct" : correct
                        }
                        if correct == 1:

                            score = score + (0.5/len(trueLabels))
                    self.df.loc[index, "score"] = score
                        
                    self.df.loc[index, "labelsPredicted"] = json.dumps(labels)

                
                

    def __saveFile(self):
        self.df = self.df.drop(["id","classNamePredicted","labelsPredicted","score"], axis=1, errors='ignore')
        self.df.to_csv(self.fileName, columns=["command", "className", "labels"])

    def add(self, command:str, className:str, labels:str):
        nouvelle_ligne = pd.DataFrame({'command' : [command], 'className' : [className], 'labels' : [labels]})
        if self.df is not None:
            self.df = pd.concat([self.df, nouvelle_ligne], ignore_index=True)
        else:
            self.df = nouvelle_ligne
        self.__saveFile()

    def getAll(self):
        return self.df
    
    def get(self, id:int):
        return self.df.iloc[id].to_dict()
    
    def update(self, id:int, command:str, className:str, labels:str):
        self.df.loc[id, 'command'] = command
        self.df.loc[id, 'className'] = className
        self.df.loc[id, 'labels'] = labels

        self.__saveFile()

    def delete(self, id:int):
        self.df = self.df.drop(id)
        self.__saveFile()