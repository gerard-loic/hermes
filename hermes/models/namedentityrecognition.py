import spacy
from spacy.training.example import Example
from spacy.util import minibatch
from spacy.lookups import Lookups
import random
from hermes.data.dataloader import DataLoader
from hermes.utils.log import Log


class NamedEntityRecognition:
    def __init__(self, envName:str):
        self.envName = envName

    def train(self):
        Log.write("Start Named Entity Recognition model training...")
        
        new_labels = ["notation", "essai"]
        
        dl = DataLoader(envName=self.envName)
        train_data = dl.getDataForNer(randomization=False)
        self.validate_training_data(train_data)
        nlp = spacy.load("fr_core_news_lg")

        if 'ner' not in nlp.pipe_names:
            ner = nlp.add_pipe('ner', last=True)
        else:
            ner = nlp.get_pipe('ner')

        # Ajouter les labels explicitement
        for label in new_labels:
            ner.add_label(label)
        
        # Vérifier aussi les labels dans les données
        for text, annotations in train_data:
            if "entities" in annotations:
                for ent in annotations['entities']:
                    if ent[2] not in ner.labels:
                        ner.add_label(ent[2])

        other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
        
        with nlp.disable_pipes(*other_pipes):
            # Initialiser l'optimiseur correctement
            optimizer = nlp.initialize()
            
            # AUGMENTER le nombre d'époques et ajuster le dropout
            epochs = 50  # Augmenté de 30 à 50
            batch_size = 8  # RÉDUIRE la taille des batchs (était 128)
            
            for epoch in range(epochs):
                random.shuffle(train_data)
                losses = {}
                batches = minibatch(train_data, size=batch_size)
                
                for batch in batches:
                    examples = []
                    for text, annotations in batch:
                        doc = nlp.make_doc(text)
                        example = Example.from_dict(doc, annotations)
                        examples.append(example)
                    
                    # Réduire le dropout pour mieux apprendre
                    nlp.update(examples, drop=0.2, losses=losses, sgd=optimizer)
                
                Log.write(f'Epoch : {epoch + 1}, Loss : {losses}')

        Log.write(f"Sauvegarde du modèle dans environnements/{self.envName}/models/ner")
        nlp.to_disk(f"environnements/{self.envName}/models/ner")
        Log.write("Modèle sauvegardé !")

    def validate_training_data(self, train_data):
        """Vérifie la qualité des annotations"""
        multi_entity_count = 0
        
        for text, annotations in train_data:
            entities = annotations.get('entities', [])
            if len(entities) > 1:
                multi_entity_count += 1
            
            # Vérifier les chevauchements
            sorted_ents = sorted(entities, key=lambda x: x[0])
            for i in range(len(sorted_ents) - 1):
                if sorted_ents[i][1] > sorted_ents[i+1][0]:
                    Log.write(f"⚠️ Chevauchement détecté : {text}")
        
        Log.write(f"Phrases avec plusieurs entités : {multi_entity_count}/{len(train_data)}")

    def load(self):
        Log.write("Load Named Entity Recognition Model...")

        self.model = spacy.load(f"environnements/{self.envName}/models/ner")

    def analyseCommand(self, command):
        doc = self.model(command)

        arguments = {}
        for ent in doc.ents:
            arguments[ent.label_] = ent.text

        return arguments
        #print([(ent.text, ent.label_) for ent in doc.ents])
