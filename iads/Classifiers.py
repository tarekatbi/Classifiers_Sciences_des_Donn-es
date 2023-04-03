# -*- coding: utf-8 -*-

"""
Package: iads
File: Classifiers.py
Année: LU3IN026 - semestre 2 - 2022-2023, Sorbonne Université
"""

# Classfieurs implémentés en LU3IN026
# Version de départ : Février 2023
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
# ------------------------ A COMPLETER :

# Recopier ici la classe Classifier (complète) du TME 2
class Classifier:
    """ Classe (abstraite) pour représenter un classifieur
        Attention: cette classe est ne doit pas être instanciée.
    """
    
    def __init__(self, input_dimension):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension de la description des exemples
            Hypothèse : input_dimension > 0
        """
        raise NotImplementedError("Please Implement this method")
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        raise NotImplementedError("Please Implement this method")
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        raise NotImplementedError("Please Implement this method")
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        raise NotImplementedError("Please Implement this method")

    def accuracy(self, desc_set, label_set):
        """ Permet de calculer la qualité du système sur un dataset donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        acc=0
        for i in range(0,len(desc_set)):
            if(self.predict(desc_set[i]) == label_set[i]):
                acc=acc+1
        
        return acc/len(desc_set)


# ------------------------ A COMPLETER : DEFINITION DU CLASSIFIEUR PERCEPTRON

class ClassifierPerceptron(Classifier):
    """ Perceptron de Rosenblatt
    """
    def __init__(self, input_dimension, learning_rate, init):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples (>0)
                - learning_rate (par défaut 0.01): epsilon
                - init est le mode d'initialisation de w: 
                    - si True (par défaut): initialisation à 0 de w,
                    - si False : initialisation par tirage aléatoire de valeurs petites
        self.input_dimension = input_dimension
        self.learning_rate = learning_rate
        self.init = init
        if (init):
          self.w = np.zeros(input_dimension)
        else:
            self.w = (2*np.random.uniform() -1)*0.0001
        #raise NotImplementedError("Please Implement this method")"""
        self.input_dimension = input_dimension
        self.learning_rate = learning_rate
        if init==True:
            self.w = np.zeros(self.input_dimension)
        else:
            self.w = np.random.randn(self.input_dimension)*0.01
            print(self.w)
        self.old_w = self.w.copy()
        self.allw =[self.w.copy()] # stockage des premiers poids
    
    def get_allw(self):
        return self.allw
        
    def train_step(self, desc_set,label_set):
        """ Réalise une unique itération sur tous les exemples du dataset
            donné en prenant les exemples aléatoirement.
            Arguments:
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
        """
        # Mélange des données
        idxs = np.arange(desc_set.shape[0])
        np.random.shuffle(idxs)
        desc_set = desc_set[idxs]
        label_set = label_set[idxs]
        
        # Pour chaque exemple
        for i in range(desc_set.shape[0]):
            # Prédiction
            x = desc_set[i]
            y = label_set[i]
            y_pred = self.predict(x)
            
            # Mise à jour du poids
            if y*y_pred <= 0:
                self.w += self.learning_rate*y*x
                self.allw.append(self.w.copy())
        #raise NotImplementedError("Please Implement this method")
     
    def train(self, desc_set, label_set, nb_max=100, seuil=0.001):
        """ Apprentissage itératif du perceptron sur le dataset donné.
            Arguments:
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
                - nb_max (par défaut: 100) : nombre d'itérations maximale
                - seuil (par défaut: 0.001) : seuil de convergence
            Retour: la fonction rend une liste
                - liste des valeurs de norme de différences
        """     
        diffs = []
        for epoch in range(nb_max):
            # Entrainement d'une étape
            self.train_step(desc_set, label_set)
            
            # Calcul de la différence
            diff_norm = np.linalg.norm(self.w-self.old_w)
            diffs.append(diff_norm)
            
            # Si convergence, arrêt
            if diff_norm < seuil:
                break
                
        return diffs
        #raise NotImplementedError("Please Implement this method")
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        return np.dot(x,self.w)
        #raise NotImplementedError("Please Implement this method")
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        if self.score(x)>=0:
            return 1
        else: 
            return -1

        #raise NotImplementedError("Please Implement this method")


# ------------------------ A COMPLETER :

# ------------------------ A COMPLETER :

class ClassifierKNN(Classifier):
    """ Classe pour représenter un classifieur par K plus proches voisins.
        Cette classe hérite de la classe Classifier
    """

    # ATTENTION : il faut compléter cette classe avant de l'utiliser !
    
    def __init__(self, input_dimension, k):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
                - k (int) : nombre de voisins à considérer
            Hypothèse : input_dimension > 0
        """
        self.input_dimension=input_dimension
        self.k=k
        self.desc=[]
        self.label=[]
        #raise NotImplementedError("Please Implement this method")
        

    
    def predict(self, x):
        """ rend la prediction sur x (-1 ou +1)
            x: une description : un ndarray
        """
        if self.score(x)>=0:
            return 1
        else: 
            return -1
        

    def score(self,x):
        """ rend la proportion de +1 parmi les k ppv de x (valeur réelle)
            x: une description : un ndarray
        """
        dist = np.linalg.norm(self.desc-x, axis=1)
        argsort = np.argsort(dist)
        score = np.sum(self.label[argsort[:self.k]] == 1)
        return 2 * (score/self.k -0.5)
        raise NotImplementedError("Please Implement this method")

    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        self.desc = desc_set
        self.label = label_set
        #raise NotImplementedError("Please Implement this method")

# ------------------------ A COMPLETER :
class ClassifierLineaireRandom(Classifier):
    """ Classe pour représenter un classifieur linéaire aléatoire
        Cette classe hérite de la classe Classifier
    """
    
    def __init__(self, input_dimension):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension de la description des exemples
            Hypothèse : input_dimension > 0
        """
        self.input_dimension = input_dimension
        v = np.random.uniform(low=-1, high=1, size=self.input_dimension)
        self.w = v / np.linalg.norm(v)
        self.desc=[]
        self.label=[]
        #raise NotImplementedError("Please Implement this method")
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """      
        print("Pas d'apprentissage pour ce classifier ! \n")
        self.desc_set = desc_set
        self.label_set = label_set 
        #raise NotImplementedError("Please Implement this method")
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        return np.dot(x,self.w)
        #raise NotImplementedError("Please Implement this method")
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        score = self.score(x)
        if score>0:
            return 1
        else: 
            return -1
        #raise NotImplementedError("Please Implement this method")


# ------------------------ A COMPLETER :

# Remarque : quand vous transférerez cette classe dans le fichier classifieur.py 
# de votre librairie, il faudra enlever "classif." en préfixe de la classe ClassifierPerceptron:

# ------------------------ A COMPLETER :

# Remarque : quand vous transférerez cette classe dans le fichier classifieur.py 
# de votre librairie, il faudra enlever "classif." en préfixe de la classe ClassifierPerceptron:

class ClassifierPerceptronBiais(ClassifierPerceptron):
    """ Perceptron de Rosenblatt avec biais
        Variante du perceptron de base
    """
    def __init__(self, input_dimension, learning_rate=0.01, init=True):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples (>0)
                - learning_rate (par défaut 0.01): epsilon
                - init est le mode d'initialisation de w: 
                    - si True (par défaut): initialisation à 0 de w,
                    - si False : initialisation par tirage aléatoire de valeurs petites
        """
        # Appel du constructeur de la classe mère
        super().__init__(input_dimension, learning_rate, init)
        # Affichage pour information (décommentez pour la mise au point)
        #print("Init perceptron biais: w= ",self.w," learning rate= ",learning_rate)
        
    def train_step(self, desc_set, label_set):
        """ Réalise une unique itération sur tous les exemples du dataset
            donné en prenant les exemples aléatoirement.
            Arguments:
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
        """        
        idxs = np.arange(desc_set.shape[0])
        np.random.shuffle(idxs)
        desc_set = desc_set[idxs]
        label_set = label_set[idxs]
        
        # Pour chaque exemple
        for i in range(desc_set.shape[0]):
            # Prédiction
            x = desc_set[i]
            y = label_set[i]
            y_pred = self.score(x)
            
            # Mise à jour du poids
            if y*y_pred < 1:
                #self.w += self.learning_rate*y*x
                self.w = self.w + self.learning_rate*((y-y_pred)*x)
                self.allw.append(self.w.copy())
        # Ne pas oublier d'ajouter les poids à allw avant de terminer la méthode
        #raise NotImplementedError("Vous devez implémenter cette méthode !")    
# ------------------------ 


