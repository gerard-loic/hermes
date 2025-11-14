from hermes.utils.log import Log
from hermes.data.dataloader import DataLoader
from hermes.models.classifier import Classifier
from hermes.models.namedentityrecognition import NamedEntityRecognition
import json

class Hermes:
    def __init__(self, envName:str):
        Log.write(
        """
=============================================================
=   Hermes - AI natural commands recognizer                 =
=   Author : Loïc Gérard                                    =
=   Version : 0.0.1                                         =
=   Published under GNU open-source licence                 =
=============================================================
        """
        )
        Log.write(f"Environnement {envName} initializd")
        self.envName = envName
        self.trained = False
        self.classifier = None
        self.ner = None

    def getEnvName(self):
        return self.envName
    
    def isTrained(self):
        return self.trained
    
    def train(self):
        #classifierModel = Classifier(self.getEnvName())
        #classifierModel.train()

        nerModel = NamedEntityRecognition(self.getEnvName())
        nerModel.train()

    def analyseCommand(self, command:str):
        self.__initializeModels()

        #Command classification
        intent, confidence = self.classifier.analyseCommand(command)

        #Get arguments
        arguments = self.ner.analyseCommand(command)

        out = {
            "action" : intent,
            "confidence" : confidence,
            "arguments" : arguments
        }

        
        return out
    
    def __initializeModels(self):
        if self.classifier == None:
            self.classifier = Classifier(envName=self.getEnvName())
            self.classifier.load()
        if self.ner == None:
            self.ner = NamedEntityRecognition(envName=self.getEnvName())
            self.ner.load()

