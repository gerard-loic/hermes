import json
from hermes.utils.log import Log
from hermes.agent.argument import Argument

class Agent:
    def __init__(self, name:str, envName:str):
        self.name = name
        self.envName = envName
        self.arguments = {}
        self.__loadConfig()

    def __loadConfig(self):
        dataFile = f"environnements/{self.envName}/rawdata/{self.name}/agent.json"
        try:
            with open(dataFile, 'r', encoding='utf-8') as fichier:
                conf = json.load(fichier)
        except FileNotFoundError:
            Log.write(f"Data file {dataFile} not found !")
        except json.JSONDecodeError as e:
            Log.write(f"Erreur de parsing JSON: {e}")

        for arg in conf["arguments"]:
            self.arguments[arg] = Argument(arg, conf["arguments"][arg])
        self.acknoledged = conf["acknoledged"]
        self.notAcknoledged = conf["not-acknoledged"]
        Log.write(f"Agent {self.name} loaded on environnement {self.envName}")

    def checkOutput(self, output):

        score = 1

        #Vérification que les arguments en sortie sont attendus
        if 1 == 2:
            args = []
            for arg in output["arguments"]:
                args.append(arg)

                for arg in args:
                    r = self.__checkArgument(arg, output["arguments"][arg])
                    if r is False:
                        del output["arguments"][arg]
        
        #Vérification si des arguments sont attendus en sortie mais ne sont pas présents
        argumentsNotPresent = []
        countTotal = 0
        for arg in self.arguments:
            if self.arguments[arg].isRequired():
                countTotal = countTotal+1
                if arg not in output["arguments"]:
                    argumentsNotPresent.append(arg)
        
        if len(argumentsNotPresent) > 0:
            #Réponse en précisant les arguments manquants
            answer = self.notAcknoledged
            replace = ""
            if len(argumentsNotPresent) == 1:
                replace = str(self.arguments[argumentsNotPresent[0]].getNotAcknoledgedLabel())
            else:
                replace = str(argumentsNotPresent)
                #replace = ", ".join(str(x) for x in self.arguments[argumentsNotPresent[:-1]].getNotAcknoledgedLabel()) + " et " + str(self.arguments[argumentsNotPresent[-1]].getNotAcknoledgedLabel())
            output["answer"] = answer.replace("%arguments%", replace)

            score = 1 - (len(argumentsNotPresent)/countTotal*0.5)
        else:
            #Réponse tout est OK
            answer = self.acknoledged
            for arg in self.arguments:
                if arg in output["arguments"]:
                    answer = answer.replace(f"%{arg}%", output["arguments"][arg])
            output["answer"] = answer

        
        if output["action"] == "default":
            score = 0

        output["capability"] = score

        return output


    def __checkArgument(self, argumentName:str, argumentValue:str):
        #Vérifie que c'est un argument attendu
        if argumentName in self.arguments:
            return True
        return False
