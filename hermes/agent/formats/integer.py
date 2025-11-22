class Integer:
    def __init__(self, config:dict={}):
        self.config = config

    @staticmethod
    def checkFormat(self, value:str):
        try:
            int(value)
            return True
        except ValueError:
            return False