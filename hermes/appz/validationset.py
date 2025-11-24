from flask import render_template, jsonify
from hermes.data.validationset import ValidationSet as DataValidationSet
from hermes.data.dataloader import DataLoader

class ValidationSet:
    def __init__(self, envName:str):
        self.envName = envName
        self.vs = None

    def render(self, score:bool=False):
        self.__loadData(score=score)
        return render_template('validationset.html', data=self.vs.getAll(), score=score)

    def exec_getCommand(self, id:int):
        self.__loadData()
        return self.vs.get(id)
    
    def exec_putCommand(self, id:int, data:dict):
        self.__loadData()
        self.vs.update(id=id, command=data["command"], className=data["className"], labels=data["labels"])
        return jsonify({'success': True})

    def exec_postCommand(self, data:dict):
        self.__loadData()
        self.vs.add(command=data["command"], className=data["className"], labels=data["labels"])
        return jsonify({'success': True})
    
    def exec_deleteCommand(self, id:int):
        self.__loadData()
        self.vs.delete(id=id)
        return jsonify({'success': True})
    

    def exec_getClasses(self):
        dl = DataLoader(envName=self.envName)
        return jsonify({'classes': dl.getClasses()})
    
    def __loadData(self, score:bool=False):
        if not self.vs:
            self.vs = DataValidationSet(self.envName, score=score)
