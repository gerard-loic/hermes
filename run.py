from fastapi import FastAPI
from pydantic import BaseModel
from hermes.environnements import Environnements
from hermes.core import Hermes
import uvicorn
from typing import Optional

#Initialize Hermes environnements handler
Environnements.init()

app = FastAPI()

class CommandRequest(BaseModel):
    command: str
    environnementName:str
    sessionUID: Optional[str] = None 

@app.post("/command")
def post_command(request: CommandRequest):
    hermes = Environnements.getHermes(envName=request.environnementName)
    
    if request.sessionUID is not None:
        out = hermes.analyseCommand(request.command, request.sessionUID)
    else:
        out = hermes.analyseCommand(request.command)

    return out

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)