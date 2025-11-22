class Varchar:
    def __init__(self, config:dict={}):
        self.config = config

    @staticmethod
    def checkFormat(self, value:str):
        return True