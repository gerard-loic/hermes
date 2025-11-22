from hermes.utils.log import Log
from hermes.data.dataloader import DataLoader
from hermes.models.classifier import Classifier
from hermes.models.namedentityrecognition import NamedEntityRecognition
from hermes.utils.climanager import CliManager
from hermes.data.sessionmemory import SessionMemory
from hermes.agent.agentsmanager import AgentsManager
from hermes.data.config import Config
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
        Log.write(f"Environnement {envName} initialized !")
        self.envName = envName
        self.trained = False
        self.classifier = None
        self.ner = None
        self.ready = False

        #Initialisation de la configuration globale
        Config.init()

        #Initalisation de la gestion mémoire
        SessionMemory.init()
        Log.write("Temporary session memory manager initialized !")

        #Initialisation de la gestion des agents
        AgentsManager.init(envName=envName)


       

    def console(self, args:list=[]):
        CliManager.cmd(args, self)

    def getEnvName(self):
        return self.envName
    
    def isTrained(self):
        return self.trained
    
    def train(self, trainClassifierModel:bool=True, trainNerModel:bool=True):
        if trainClassifierModel:
            classifierModel = Classifier(self.getEnvName())
            classifierModel.train()

        if trainNerModel:
            nerModel = NamedEntityRecognition(self.getEnvName())
            nerModel.train()

    def analyseCommand(self, command:str, sessionUID:str=None):
        self.__initializeModels()

        if sessionUID is None:
            return self.__newCommand(command)
        else:
            return self.__continueCommand(command, sessionUID)
    
    def __newCommand(self, command:str, sessionUID:str=None):
        #Command classification
        intent, confidence = self.classifier.analyseCommand(command)

        #Get arguments
        arguments = self.ner.analyseCommand(command)

        #get Agent
        agent = AgentsManager.getAgent(intent)

        #session
        if sessionUID is None:
            sessionUID = SessionMemory.createSession({})

        #Construction de la sortie
        out = {
            "action" : intent,
            "arguments" : arguments,
            "answer" : "",
            "confidence" : confidence,
            "capability" : 1,
            "sessionUID" : sessionUID
        }

        if agent != None:
            #Vérification de la sortie via un agent
            out = agent.checkOutput(out)

        #Mise à jour des données en cache
        SessionMemory.updateSession(sessionUID, out)
        Log.write("MISE EN CACHE ")
        Log.write(sessionUID)

        out = SessionMemory.getSession(sessionId=sessionUID)
        Log.write(out)
        
        return out
    
    def __continueCommand(self, command:str, sessionUID:str):
        Log.write("SESSION UID :")
        Log.write(sessionUID)
        #On récupère la session
        out = SessionMemory.getSession(sessionId=sessionUID)
        if out is None:
            #Cas particulier : timeout @TODO
            Log.write("Session timeout !")

        Log.write(out)

        if out["action"] == "default":
            #Cas particulier, on avait pas réussi à identifier une commande
            return self.__newCommand(command, sessionUID)
        
        #On tente de compléter la commande avec les nouvelles données

        #Get arguments
        out["arguments"] = out["arguments"] | self.ner.analyseCommand(command)

        #get Agent
        agent = AgentsManager.getAgent(out["action"])

        if agent != None:
            #Vérification de la sortie via un agent
            out = agent.checkOutput(out)

        #Mise à jour des données en cache
        SessionMemory.updateSession(sessionUID, out)

        return out
    
    def __initializeModels(self):
        if self.classifier == None:
            Log.write(f"Initialize models in environnement {self.envName}")
            self.classifier = Classifier(envName=self.getEnvName())
            self.classifier.load()
        if self.ner == None:
            self.ner = NamedEntityRecognition(envName=self.getEnvName())
            self.ner.load()
        if self.ready == False:
            Log.write(f"Initialize agents in environnement {self.envName}")
            dl = DataLoader(envName=self.envName)
            classes = dl.getClasses()
            for c in classes:
                agent = AgentsManager.addAgent(name=c)
        self.ready = True

