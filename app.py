from flask import Flask, render_template, request, jsonify, Response
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # Backend non-interactif pour serveur
import base64
import io
import sys
sys.path.append('../')
from hermes.data.datastats import DataStats
from hermes.appz.donneesentrainement import DonneesEntrainement
from hermes.appz.entrainementmodeles import EntrainementModeles
from hermes.appz.classifiertrainer import ClassifierTrainer
from hermes.appz.testeurcommandes import TesteurCommandes
from hermes.appz.validationset import ValidationSet
import threading
import time
import queue
import json
from hermes.appz.trainingqueue import TrainingQueue

app = Flask("Hermes : administration console")


######################################################################################


# Dictionnaire pour stocker les queues de messages par session
TrainingQueue.init()

######################################################################################

@app.route('/')
def index():
    return render_template('index.html')

@app.template_filter('fromjson')
def fromjson_filter(value):
    return json.loads(value)

######################################################################################

@app.route('/donneesentrainement')
def donneesentrainement():
    r = DonneesEntrainement(envName="default")
    return r.render()

######################################################################################

@app.route('/testeurcommandes')
def testeurcommandes():
    r = TesteurCommandes(envName="default")
    return r.render()

######################################################################################

@app.route('/validationset')
def validationset():
    score = request.args.get('score')

    r = ValidationSet(envName="default") 
    if(score):
        return r.render(score=True)
    else:
        return r.render()

@app.route('/validationset', methods=['POST'])
def validationset_post():
    data = request.get_json()
    r = ValidationSet(envName="default")
    return r.exec_postCommand(data=data)
    try:
        data = request.get_json()
        r = ValidationSet(envName="default")
        return r.exec_postCommand(data=data)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/validationset/<int:id>', methods=['GET'])
def validationset_get(id):
    r = ValidationSet(envName="default")
    return r.exec_getCommand(id)

@app.route('/validationset/<int:id>', methods=['PUT'])
def validationset_put(id):
    try:
        data = request.get_json()
        
        r = ValidationSet(envName="default")
        return r.exec_putCommand(id, data)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/validationset/<int:id>', methods=['DELETE'])
def validationset_delete(id):
    try:
        r = ValidationSet(envName="default")
        return r.exec_deleteCommand(id)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})
    
@app.route('/validationset/classes', methods=['GET'])
def validationset_get_classes():
    r = ValidationSet(envName="default")
    return r.exec_getClasses()


@app.route('/testeurcommandes/commande', methods=['POST'])
def testeurcommandes_commande():
    r = TesteurCommandes(envName="default")
    return r.exec_commande(request.get_json())

@app.route('/entrainementmodeles')
def entrainementmodeles():
    r = EntrainementModeles(envName="default")
    return r.render()

@app.route('/entrainementmodeles/classification/train', methods=['POST'])
def entrainementmodeles_classification_train():
    r = ClassifierTrainer(envName="default")
    return r.render(request.get_json())
    

@app.route('/entrainementmodeles/classification/train/<session_id>')
def entrainementmodeles_classification_train_session(session_id):
    """Stream SSE pour les mises à jour de progression"""
    def generate():
        q = TrainingQueue.get(session_id)
        if not q:
            yield f"data: {json.dumps({'status': 'error', 'message': 'Session non trouvée'})}\n\n"
            return
        
        while True:
            try:
                message = q.get(timeout=30)
                if message == 'DONE':
                    # Nettoyer la queue
                    TrainingQueue.delete(session_id)
                    break
                yield f"data: {message}\n\n"
            except queue.Empty:
                # Envoyer un heartbeat pour maintenir la connexion
                yield f"data: {json.dumps({'status': 'heartbeat'})}\n\n"
    
    return Response(generate(), mimetype='text/event-stream')

if __name__ == '__main__':
    # Accessible depuis Windows sur http://localhost:5000
    app.run(host='0.0.0.0', port=5000, debug=True)
