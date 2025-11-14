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
        #On définit les nouveaux labels
        new_labels = [ 
            "NOTATION"
        ]

        #On récupère les données et on prépare
        dl = DataLoader(envName=self.envName)
        train_data = dl.getDataForNer(randomization=False)
        nlp = spacy.load("fr_core_news_lg")

        #Ajouter ner dans le pipeline
        #Cette partie vérifie si le pipeline spaCy contient déjà un composant NER :
        #Si non : on ajoute un nouveau composant NER vierge au pipeline
        #Si oui : on récupère le composant NER existant (pour ne pas le dupliquer)

        if 'ner' not in nlp.pipe_names:
            ner = nlp.add_pipe('ner')
        else:
            ner = nlp.get_pipe('ner')


        #Cette boucle parcourt vos données d'entraînement pour déclarer tous les types d'entités possibles :
        #train_data contient des paires (texte, annotations)
        #annotations['entities'] est une liste de tuples comme (début, fin, label)
        #ent[2] récupère le label (par exemple "PERSONNE", "LIEU", "ORGANISATION")
        #ner.add_label(ent[2]) déclare ce type d'entité au modèle
        #Pourquoi c'est nécessaire ? Le modèle doit connaître à l'avance tous les types d'entités qu'il devra prédire.

        for data_sample, annotations in train_data:
            for ent in annotations['entities']:
                if ent[2] not in ner.labels:
                    ner.add_label(ent[2])


        #Disable other pipes (such as classification, POS Tagging, etc)
        other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
        
        with nlp.disable_pipes(*other_pipes):
            optimizer = nlp.resume_training()
            epochs = 30
            for epoch in range(epochs):
                random.shuffle(train_data) # shuffling the dataset for each epoch 
                losses = {}
                batches = minibatch(train_data, size = 128)
                for batch in batches:
                    examples = []
                    for text, annotations in  batch:
                        doc = nlp.make_doc(text)
                        example = Example.from_dict(doc, annotations)
                        examples.append(example)
                    nlp.update(examples, drop = 0.15, losses = losses)
                Log.write(f'Epoch : {epoch + 1}, Loss : {losses}')

        Log.write(f"Sauvegarde du modèle dans environnements/{self.envName}/models/ner")
        nlp.to_disk(f"environnements/{self.envName}/models/ner")
        Log.write("Modèle sauvegardé !")

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
