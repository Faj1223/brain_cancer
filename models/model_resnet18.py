# Importation de la bibliothèque PyTorch
import torch

# Importation du module pour créer et gérer des réseaux de neurones
import torch.nn as nn

# Importation des modèles pré-entraînés disponibles dans torchvision
from torchvision import models

# Définition d'une classe pour un modèle personnalisé basé sur ResNet
class ResNetBinaryClassifier(nn.Module):  # Hérite de nn.Module, une base pour tous les modèles PyTorch
    def __init__(self, pretrained=True):  # Initialisation de la classe avec un paramètre pour utiliser un modèle pré-entraîné
        # Appel du constructeur de la classe parente nn.Module pour initialiser les fonctionnalités de base
        super(ResNetBinaryClassifier, self).__init__()
        
        # Charger un modèle ResNet18 pré-entraîné sur ImageNet (si pretrained=True)
        self.resnet = models.resnet18(pretrained=pretrained)
        
        # Extraire le nombre de caractéristiques en sortie de la dernière couche du ResNet
        num_features = self.resnet.fc.in_features
        
        # Remplacer la dernière couche entièrement connectée (fc) du ResNet pour une sortie unique (classification binaire)
        # La sortie est un seul score pour déterminer si l'image appartient à la classe 0 ou 1
        self.resnet.fc = nn.Linear(num_features, 1)
        
        # Ajouter une fonction sigmoïde pour transformer le score en probabilité entre 0 et 1
        self.sigmoid = nn.Sigmoid()

    # Définir la méthode de propagation avant (forward) pour le modèle
    def forward(self, x):
        # Passer les données d'entrée dans le modèle ResNet
        x = self.resnet(x)
        
        # Appliquer la fonction sigmoïde pour convertir le score brut en probabilité
        return self.sigmoid(x)
    

class CustomResNetClassifier(ResNetBinaryClassifier):  
    def __init__(self, pretrained=True):
        super(CustomResNetClassifier, self).__init__(pretrained=pretrained)
        
        # Modifier la tête du réseau pour les sorties à 4 classes
        self.resnet.fc = nn.Sequential(
            self.resnet.fc,                # Couche FC d'origine
            nn.Linear(1, 256),             # Nouvelle couche FC avec 256 unités
            nn.ReLU(),                     # Activation ReLU
            nn.Linear(256, 128),           # Deuxième couche FC avec 128 unités
            nn.ReLU(),                     # Activation ReLU
            nn.Linear(128, 4),             # Dernière couche FC avec 4 unités
            nn.Softmax(dim=1)              # Activation Softmax pour probabilités sur 4 classes
        )




   

