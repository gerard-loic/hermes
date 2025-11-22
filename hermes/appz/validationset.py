from flask import render_template
from hermes.data.validationset import ValidationSet as DataValidationSet

class ValidationSet:
    def __init__(self, envName:str):
        self.envName = envName

    def render(self):
        vs = DataValidationSet(self.envName)
        return render_template('validationset.html', data=vs.getAll())

    def exec_getCommand(self, id:int):
        vs = DataValidationSet(self.envName)
        return vs.get(id)
