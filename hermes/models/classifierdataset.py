from hermes.utils.log import Log
import torch
from torch.utils.data import Dataset


# Dataset personnalisé (hérise de la classe Dataset de PyTorch)

class ClassifierDataset(Dataset):
    
    def __init__(self, texts, labels, tokenizer, max_length=64):
        """
        Constructeur de la classe.
        
        Args:
            texts (list): Liste de chaînes de caractères à classifier
            labels (list): Liste d'entiers représentant les classes cibles
            tokenizer: Tokenizer de Transformers (ex: CamembertTokenizer)
            max_length (int): Longueur maximale des séquences après tokenization
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        """
        Méthode obligatoire pour Dataset.
        Retourne le nombre total d'échantillons dans le dataset.
        Utilisée par DataLoader pour connaître la taille du dataset.
        """
        return len(self.texts)
    
    def __getitem__(self, idx):
        """
        Méthode obligatoire pour Dataset.
        Retourne un échantillon spécifique du dataset à l'index donné.
        
        Args:
            idx (int): Index de l'échantillon à récupérer (0 à len-1)
            
        Returns:
            dict: Dictionnaire contenant les tenseurs nécessaires pour l'entraînement
        """

        # 1. Récupération des données brutes à l'index idx
        text = self.texts[idx]
        label = self.labels[idx]
        
        # 2. Tokenization du texte
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 3. Préparation du dictionnaire de sortie
        return {
            # input_ids: séquence de tokens numériques [1, 64] → [64]
            'input_ids': encoding['input_ids'].flatten(),

            # attention_mask: masque indiquant les tokens réels vs padding [1, 64] → [64]
            # 1 = token réel, 0 = padding
            'attention_mask': encoding['attention_mask'].flatten(),

            # label: classe cible sous forme de tenseur long (requis pour CrossEntropyLoss)
            'label': torch.tensor(label, dtype=torch.long)
        }