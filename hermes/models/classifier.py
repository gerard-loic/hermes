from hermes.utils.log import Log
from hermes.data.dataloader import DataLoader as HermesDataLoader
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import CamembertTokenizer, CamembertForSequenceClassification
from torch.optim import AdamW  # AdamW est maintenant dans torch.optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import numpy as np
from hermes.models.classifierdataset import ClassifierDataset
from tqdm import tqdm
from hermes.models.trainstats import TrainStats
from hermes.appz.trainingqueue import TrainingQueue
from hermes.appz.entrainementmodeles import EntrainementModeles

class Classifier:
    def __init__(self, envName:str, trainingQueueSessionId:str=None):
        print("Classifier")
        self.envName = envName
        self.trainingQueueSessionId = trainingQueueSessionId

    def train(self, epochs=5):
        Log.write("Start classifier model training...")
        

        TrainingQueue.sendMessage(
            self.trainingQueueSessionId,
            {
                'status': 'training',
                'message' : "Initialisation de l'entraînement en cours..."
            }
        )

        #Statistiques d'entrainement
        self.trainStats = TrainStats(envName=self.envName, model="classifier")
        self.trainStats.removeTrainStat()
        
        #Définition du device utilisé
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        Log.write(f"Utilisation de: {device}")
        
        #Dataloader
        dl = HermesDataLoader(envName="default")

        # Mapping des labels vers des indices
        label_to_id, id_to_label = dl.getClassesDictionnaryForClassifier()
        labels, texts = dl.getDataForClassifier()
        labels = [label_to_id[item] for item in labels]
            
        # Split train/test
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
            
        Log.write(f"Données d'entraînement: {len(train_texts)}")
        Log.write(f"Données de test: {len(test_texts)}")

        # Chargement du tokenizer et du modèle
        Log.write("Chargement de CamemBERT...")
        tokenizer = CamembertTokenizer.from_pretrained('camembert-base')
        model = CamembertForSequenceClassification.from_pretrained(
            'camembert-base',
            num_labels=len(label_to_id)
        )
        model.to(device)

        # Création des datasets
        train_dataset = ClassifierDataset(train_texts, train_labels, tokenizer)
        test_dataset = ClassifierDataset(test_texts, test_labels, tokenizer)
            
        train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=8)

        # Entraînement
        optimizer = AdamW(model.parameters(), lr=2e-5)
            
        Log.write("Entraînement du modèle...")
        self.__trainModel(train_dataloader, test_dataloader, model, optimizer, device, epochs=epochs)


        # Évaluation
        Log.write("Évaluation sur le jeu de test...")
        predictions, true_labels = self.__evaluateModel(test_dataloader, model, device)

        cm = confusion_matrix(true_labels, predictions)
          
        Log.write("Rapport de classification:")
        Log.write(classification_report(
            true_labels,
            predictions,
            target_names=list(label_to_id.keys())
        ))

        self.trainStats.writeClassificationReport(
            classification_report(
                true_labels,
                predictions,
                target_names=list(label_to_id.keys()),
                output_dict=True
            )
        )

        self.trainStats.writeConfusionMatrix(cm)

        self.__saveModel(model, tokenizer)

        TrainingQueue.sendMessage(
            self.trainingQueueSessionId,
            {
                'status': 'completed',
                'message': 'Entraînement terminé avec succès !',
                'progress': 100,
                'progressGlobal' : 100,
            }
        )


    def load(self):
        Log.write("Load classifier model...")

        #Matrices de correspondance
        dl = HermesDataLoader(envName="default")

        # Mapping des labels vers des indices
        self.label_to_id, self.id_to_label = dl.getClassesDictionnaryForClassifier()

        #Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Charger le tokenizer
        self.tokenizer = CamembertTokenizer.from_pretrained(f"environnements/{self.envName}/models/classifier")

        # Charger le modèle
        self.model = CamembertForSequenceClassification.from_pretrained(f"environnements/{self.envName}/models/classifier")
        self.model.to(self.device)

        Log.write("Classifier model loaded !")

    def analyseCommand(self, command):
        """Prédit l'intention pour une phrase"""
        self.model.eval()
        
        encoding = self.tokenizer(
            command,
            add_special_tokens=True,
            max_length=64,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            pred = torch.argmax(probs, dim=1)
            confidence = probs[0][pred].item()
        
        return self.id_to_label[pred.item()], confidence

    #---------------------------------------------------------------------------------------------------------------------
        
    def __trainModel(self, train_dataloader, val_dataloader, model, optimizer, device, epochs=3):
        """Entraîne le modèle avec barre de progression"""
        model.train()
        
        
        for epoch in range(epochs):
            total_loss = 0
            correct_predictions = 0
            total_predictions = 0
            
            # Créer une barre de progression pour chaque epoch
            progress_bar = tqdm(
                train_dataloader, 
                desc=f"Epoch {epoch + 1}/{epochs}",
                unit="batch"
            )
            
            for batch in progress_bar:
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_loss += loss.item()

                # Calcul de l'accuracy
                preds = torch.argmax(outputs.logits, dim=1)
                correct_predictions += (preds == labels).sum().item()
                total_predictions += labels.size(0)
                
                loss.backward()
                optimizer.step()
                
                # Mettre à jour la barre de progression avec la loss actuelle
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{total_loss / (progress_bar.n + 1):.4f}'
                })
                TrainingQueue.sendMessage(
                    self.trainingQueueSessionId,
                    {
                        'status': 'training',
                        'epoch': (epoch+1),
                        'total_epochs': epochs,
                        'progress': round((progress_bar.n / progress_bar.total)*100),
                        'progressGlobal' : round((epoch/epochs)*100),
                        'message': f"Epoch {(epoch+1)} (loss: {loss.item():.4f} - avg_loss: {total_loss / (progress_bar.n + 1):.4f})"
                    }
                )
            
            #Metrique moyennes d'entraînement
            avg_train_loss = total_loss / len(train_dataloader)
            train_acc = correct_predictions / total_predictions


            # ========== PHASE DE VALIDATION ==========
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0

            progress_bar_val = tqdm(
                val_dataloader,
                desc=f"Epoch {epoch + 1}/{epochs} [Val]",
                unit="batch"
            )

            with torch.no_grad():
                for batch in progress_bar_val:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['label'].to(device)
                    
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    
                    val_loss += outputs.loss.item()
                    
                    preds = torch.argmax(outputs.logits, dim=1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)
                    
                    progress_bar_val.set_postfix({
                        'val_loss': f'{outputs.loss.item():.4f}'
                    })
            # Métriques moyennes de validation
            avg_val_loss = val_loss / len(val_dataloader)
            val_acc = val_correct / val_total
            #Log.write(f"Epoch {epoch + 1}/{epochs} terminée - Loss moyenne: {avg_train_loss:.4f}")

            Log.write(f"\n{'='*60}")
            Log.write(f"Epoch {epoch + 1}/{epochs} - Résumé:")
            Log.write(f"  Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f}")
            Log.write(f"  Val Loss:   {avg_val_loss:.4f} | Val Acc:   {val_acc:.4f}")
            Log.write(f"{'='*60}\n")


            #Ecriture des statistiques
            self.trainStats.writeTrainStat(
                data={
                    "epoch" : (epoch+1),
                    "train_loss" : avg_train_loss,
                    "val_loss" : avg_val_loss,
                    "train_accuracy" : train_acc,
                    "val_accuracy" : val_acc
                }
            )

            #Données à renvoyer à l'interface (si utilisation avec l'interface)
            imageResult1 = None
            if self.trainingQueueSessionId:
                r = EntrainementModeles(envName=self.envName)
                imageResult1 = r.graphClassifierTrainLoss()
                imageResult2 = r.graphClassifierTrainAccuracy()

            TrainingQueue.sendMessage(
                self.trainingQueueSessionId,
                {
                    'status': 'training',
                    'epoch': epoch,
                    'total_epochs': epochs,
                    'progress': (progress_bar.n / progress_bar.total)*100,
                    'progressGlobal' : round((epoch/epochs)*100),
                    'loss': round(avg_train_loss, 4),
                    'accuracy': round(train_acc, 4),
                    'imageResult1' : imageResult1,
                    'imageResult2' : imageResult2
                }
            )

            

        

    def __evaluateModel(self, dataloader, model, device):
        """Évalue le modèle"""
        model.eval()
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                preds = torch.argmax(outputs.logits, dim=1)
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        return predictions, true_labels
    
    def __saveModel(self, model, tokenizer):
        # Sauvegarde du modèle
        Log.write(f"Sauvegarde du modèle dans environnements/{self.envName}/models/classifier")
        model.save_pretrained(f"environnements/{self.envName}/models/classifier")
        tokenizer.save_pretrained(f"environnements/{self.envName}/models/classifier")
        Log.write("Modèle sauvegardé")


