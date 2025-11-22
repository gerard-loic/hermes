from dotenv import load_dotenv
from hermes.utils.log import Log
from pathlib import Path
import os

class Config:
    @staticmethod
    def init():
        dotenv_path = Path('.env')
        load_dotenv(dotenv_path=dotenv_path)
        Log.write(".env configuration file loaded !")

    @staticmethod
    def get(confName:str, format:str="str"):
        if format == "int":
            return int(os.getenv(confName))
        else:
            return os.getenv(confName)