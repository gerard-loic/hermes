from hermes.agent.formats.integer import Integer
from hermes.agent.formats.varchar import Varchar
from hermes.utils.log import Log

class Argument:
    def __init__(self, argumentName:str, conf:dict):
        self.notAcknoledgedLabel = conf["not-acknoledged-label"]
        self.required = conf["required"]
        self.formats = {}

        for format in conf["formats"]:
            if format == "INTEGER":
                self.formats["INTEGER"] = Integer()
            elif format == "VARCHAR":
                self.formats["VARCHAR"] = Varchar()
            else:
                Log.write(f"{argumentName} format {format} not supported !")

    def checkFormat(self, value:str):
        for format in self.formats:
            v = self.formats[format].checkFormat(value)
            if v:
                return format
        return None
    
    def isRequired(self):
        return self.required
    
    def getNotAcknoledgedLabel(self):
        return self.notAcknoledgedLabel