from hermes.core import Hermes
from flask import render_template, jsonify

class TesteurCommandes:
    def __init__(self, envName:str):
        self.envName = envName
        

    def render(self):
        TesteurCommandes.hermes = Hermes(envName=self.envName)
        return render_template('testeurcommandes.html')

    def exec_commande(self, data):
        commande = data.get("command")
        sessionUID = data.get("sessionUID", None)
        answer = TesteurCommandes.hermes.analyseCommand(command=commande, sessionUID=sessionUID)

        print(answer)
        return jsonify(answer)