from hermes.data.dataloader import DataLoader
from flask import render_template
import pandas as pd 
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # Backend non-interactif pour serveur
import base64
import io
import os
from sklearn.metrics import ConfusionMatrixDisplay

class EntrainementModeles:
    def __init__(self, envName:str):
        self.envName = envName
        self.dl = DataLoader(envName=envName)
        self.classes = self.dl.getClasses()

    def render(self):
        graphClass1 = self.graphClassifierTrainLoss()
        graphClass2 = self.graphClassifierTrainAccuracy()
        graphClass3 = self.__graphClassifierAccuracyByClasse()
        graphClass4 = self.__graphClassifierRecallByClasse()
        graphClass5 = self.__graphClassifierF1ByClasse()
        graphClass6 = self.__graphClassifierMatrix()

        return render_template('entrainementmodeles.html', graphClass1=graphClass1, graphClass2=graphClass2, graphClass3=graphClass3, graphClass4=graphClass4, graphClass5=graphClass5, graphClass6=graphClass6)

    def graphClassifierTrainLoss(self):
        
        fileName = f"environnements/{self.envName}/stats/classifier-train.csv"
        if not os.path.isfile(fileName):
            return None
        df = pd.read_csv(fileName)

        plt.figure(figsize=(10, 6))
        plt.plot(df['epoch'], df['train_loss'], marker='o', linestyle='-', linewidth=2, color="blue", label="Jeu d'entraînement")
        plt.plot(df['epoch'], df['val_loss'], marker='o', linestyle='-', linewidth=2, color="red", label="Jeu de validation")

        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title("Evolution de la loss durant l'entraînement", fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        # Sauvegarder le graphique en mémoire
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        
        # Encoder en base64 pour l'HTML
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close()  # Libérer la mémoire
        return image_base64
    
    def graphClassifierTrainAccuracy(self):
        fileName = f"environnements/{self.envName}/stats/classifier-train.csv"
        if not os.path.isfile(fileName):
            return None
        df = pd.read_csv(fileName)

        plt.figure(figsize=(10, 6))
        plt.plot(df['epoch'], df['train_accuracy'], marker='o', linestyle='-', linewidth=2, color="blue", label="Jeu d'entraînement")
        plt.plot(df['epoch'], df['val_accuracy'], marker='o', linestyle='-', linewidth=2, color="red", label="Jeu de validation")

        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title("Evolution de l'accuracy' durant l'entraînement", fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        # Sauvegarder le graphique en mémoire
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        
        # Encoder en base64 pour l'HTML
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close()  # Libérer la mémoire
        return image_base64
    
    def __graphClassifierAccuracyByClasse(self):
        df = pd.read_csv(f"environnements/{self.envName}/stats/classifier-classificationreport.csv")
        df = df[df["classe"].isin(self.classes)]

        plt.figure(figsize=(10, 6))
        plt.bar(df["classe"], df["precision"])
        plt.xlabel('Classe')
        plt.ylabel('Accuracy')
        plt.title("Accuracy - jeu d'entraînement - par classe")
        plt.tight_layout()
        
        # Sauvegarder le graphique en mémoire
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        
        # Encoder en base64 pour l'HTML
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close()  # Libérer la mémoire

        return image_base64
    
    def __graphClassifierRecallByClasse(self):
        df = pd.read_csv(f"environnements/{self.envName}/stats/classifier-classificationreport.csv")
        df = df[df["classe"].isin(self.classes)]

        plt.figure(figsize=(10, 6))
        plt.bar(df["classe"], df["recall"])
        plt.xlabel('Classe')
        plt.ylabel('Accuracy')
        plt.title("Recall - jeu d'entraînement - par classe")
        plt.tight_layout()
        
        # Sauvegarder le graphique en mémoire
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        
        # Encoder en base64 pour l'HTML
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close()  # Libérer la mémoire

        return image_base64
    
    def __graphClassifierF1ByClasse(self):
        df = pd.read_csv(f"environnements/{self.envName}/stats/classifier-classificationreport.csv")
        df = df[df["classe"].isin(self.classes)]

        plt.figure(figsize=(10, 6))
        plt.bar(df["classe"], df["f1-score"])
        plt.xlabel('Classe')
        plt.ylabel('Score f1')
        plt.title("Score f1 - jeu d'entraînement - par classe")
        plt.tight_layout()
        
        # Sauvegarder le graphique en mémoire
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        
        # Encoder en base64 pour l'HTML
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close()  # Libérer la mémoire

        return image_base64
    
    def __graphClassifierMatrix(self):
        df = pd.read_csv(f"environnements/{self.envName}/stats/classifier-confusionmatrix.csv")

        plt.figure(figsize=(10, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=df.values, display_labels=self.classes)
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Matrice de confusion")
        plt.xticks(rotation=45)

        # Sauvegarder le graphique en mémoire
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)

        
        
        # Encoder en base64 pour l'HTML
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close()  # Libérer la mémoire

        return image_base64