from hermes.appz.trainingqueue import TrainingQueue
import threading
import json
import time
from flask import Flask, render_template, request, jsonify, Response
from hermes.models.classifier import Classifier

class ClassifierTrainer:
    def __init__(self, envName:str):
        self.envName = envName

    def render(self, data):
        epochs = int(data.get('epochs', 5))
        auto_select_best = data.get('autoSelectBest', True)
        early_stop = data.get('earlyStop', True)
        patience = int(data.get('patience', 10))
        session_id = data.get('sessionId')

        session_id="A1"

        # Ici, lancez votre entraînement avec ces paramètres
        print(f"Démarrage de l'entraînement avec:")
        print(f"- Epochs: {epochs}")
        print(f"- Auto-sélection: {auto_select_best}")
        print(f"- Early stopping: {early_stop}")
        print(f"- Patience: {patience}")

        # Créer une queue pour cette session
        TrainingQueue.add(session_id)


        
        # Lancer l'entraînement dans un thread séparé
        """
        thread = threading.Thread(
            target=self.train_model,
            args=(epochs, auto_select_best, early_stop, patience, session_id)
        )
        """
        model = Classifier(envName=self.envName, trainingQueueSessionId=session_id)
        thread = threading.Thread(
            target=model.train,
            args=(epochs,)
        )

        thread.daemon = True
        thread.start()
        
        return jsonify({
            'status': 'success',
            'message': 'Entraînement démarré',
            'session_id': session_id
        })
    
    