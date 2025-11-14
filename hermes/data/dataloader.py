import pandas as pd
import json
from hermes.utils.log import Log
import random
from pathlib import Path

class DataLoader:
    def __init__(self, envName:str):
        self.envName = envName
        self.data = {}
        self.classes = []
        self.load()

    def load(self):
        #Chargement des classes
        repertoire = Path(f"environnements/{self.envName}/rawdata")

        # Lister tous les sous-répertoires
        ssRep = [item for item in repertoire.iterdir() if item.is_dir()]
        for sr in ssRep:
            self.classes.append(sr.name)
            self.data[sr.name] = {}

        for c in self.classes:
            #Chargement fichier pour classifier
            dataFile = f"environnements/{self.envName}/rawdata/{c}/data-classifier.json"
            try:
                with open(dataFile, 'r', encoding='utf-8') as fichier:
                    newData = json.load(fichier)["variations"]
                    self.data[c]["classifier"] = newData
                Log.write(f"Data file {dataFile} loaded !")
            except FileNotFoundError:
                Log.write(f"Data file {dataFile} not found !")
            except json.JSONDecodeError as e:
                Log.write(f"Erreur de parsing JSON: {e}")

            #Chargement fichier pour NER
            dataFile = f"environnements/{self.envName}/rawdata/{c}/data-ner.json"
            try:
                with open(dataFile, 'r', encoding='utf-8') as fichier:
                    newData = json.load(fichier)["variations"]
                    self.data[c]["ner"] = newData
                Log.write(f"Data file {dataFile} loaded !")
            except FileNotFoundError:
                Log.write(f"Data file {dataFile} not found !")
            except json.JSONDecodeError as e:
                Log.write(f"Erreur de parsing JSON: {e}")


    def getClasses(self):
        return self.classes
        
    def getDataForClassifier(self, randomization:bool=False, seed:int=42):
        texts = []
        labels = []
        for c in self.data:
            for l in self.data[c]["classifier"]:
                texts.append(l)
                labels.append(c)
        
        if randomization:
            random.seed(seed)
            random.shuffle(texts)
            random.shuffle(labels)

        return labels, texts
    
    def getClassesDictionnaryForClassifier(self):
        classes = self.getClasses()

        labelToId = {}
        i = 0
        for c in classes:
            labelToId[c] = i
            i = i+1

        idToLabel = {v: k for k, v in labelToId.items()}

        return labelToId, idToLabel

    def getDataForNer(self, randomization:bool=False, seed:int=42):
        data = []

        for c in self.classes:
            data.extend(self.data[c]["ner"])

        if randomization:
            random.seed(seed)
            random.shuffle(data)

        return data
        