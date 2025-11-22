import pandas as pd

class Feedback:
    def __init__(self, envName):
        self.envName = envName

    def __loadFile(self):
        fileName = f"environnements/{self.envName}/feedback.csv"