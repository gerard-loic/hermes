import pandas as pd
import os

class ValidationSet:
    def __init__(self, envName:str):
        self.envName = envName
        self.fileName = fileName = f"environnements/{self.envName}/validationset.csv"
        self.df = None
        self.__loadFile()
        

    def __loadFile(self):
        if os.path.isfile(self.fileName):
            self.df = pd.read_csv(self.fileName)

    def __saveFile(self):
        self.df.to_csv(self.fileName, columns=["command", "className", "labels"])

    def add(self, command:str, className:str, labels:dict={}):
        nouvelle_ligne = pd.DataFrame({'command' : [command], 'className' : [className], 'labels' : [labels]})
        if self.df:
            self.df = pd.concat([self.df, nouvelle_ligne], ignore_index=True)
        else:
            self.df = nouvelle_ligne
        self.__saveFile()

    def getAll(self):
        df = self.df.copy()
        df['id'] = range(len(df))
        return df
    
    def get(self, id:int):
        return self.df.iloc[id].to_dict()