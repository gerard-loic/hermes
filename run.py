from fastapi import FastAPI
from pydantic import BaseModel
from hermes.environnements import Environnements
from hermes.core import Hermes

#Initialize Hermes environnements handler
Environnements.init()

app = FastAPI()

class CommandRequest(BaseModel):
    command: str
    environnementName:str

@app.post("/command")
def post_command(request: CommandRequest):
    hermes = Environnements.getHermes(envName=request.environnementName)
    out = hermes.analyseCommand(request.command)

    return out

