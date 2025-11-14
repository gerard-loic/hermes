from hermes.utils.log import Log
from hermes.core import Hermes

class Environnements:
    @staticmethod
    def init():
        Log.write("Environnements intialization")
        Environnements.environnements = {}

    @staticmethod
    def getHermes(envName:str):
        if envName in Environnements.environnements:
            return Environnements.environnements[envName]
        hermes = Hermes(envName=envName)
        Environnements.environnements[envName] = hermes
        return hermes