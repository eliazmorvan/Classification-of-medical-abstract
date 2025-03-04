# Mantu_AI_technical_test

# Classification de compte-rendus médicaux

Le but est de prédire la condition médicale en fonction d'un compte-rendu médical. Ce projet utilise un modèle pré-entrainé BioBERT pour la classification de comptes-rendus médicaux dans différentes catégories.

Voici les catégories : [“Neoplasms”, “Digestive system diseases”, “Nervoussystem diseases”, “Cardiovasculardiseases”]

Après analyse des données et des premiers essais de modèles, j'ai supprimé la catégorie "General pathological conditions" car celle-ci est ambigüe. En effet, elle ne correspond pas à une maladie telle que les 4 autres catégories. Cela expliquait les mauvaise performances des premiers modèles sur les données complètes (accuracy ~ 60%).
Les comptes-rendus correspondant au label 'General pathological conditions' dans les données de test seront automatiquement supprimées lors de l'éxecution de la prédiction. 

## Prérequis

Assurez-vous que les outils suivants sont installés sur votre machine avant de commencer.

- Python 3.7+
- Pip (gestionnaire de paquets Python)

### Dépendances

Installez les dépendances requises via pip :

```bash
pip install -r requirements.txt
```

### Fichiers du dossier

- task.ipynb : Fichier principal contenant l'analyse des données, leur pré-traitement, l'entraînement, l'optimisation et les performances des différents modèles réalisés.
- predict.py : Module python permettant l'éxecution de la prédiction sur les données de test. Le modèle utilisé est un réseau de neurones BioBERT pré-entrainé et fine-tuné pour notre tâche de classification
- data : données utilisées pour l'entraînement

### Execution du module predict.py

Avant d'executer le module, il faut ajouter le modèle à l'environnement. Pour cela, il faut executer les blocs de code jusqu'au bloc "trainer.save_model()". L'entraînement du modèle peut prendre quelques minutes. Vous trouverez ensuite un dossier "final_model" que vous pourrez utiliser pour la prédiction.

Dans votre terminal, exécutez le script predict.py comme suit :

```bash
python predict.py --input_file <path_to_input_file.csv> --output_file <path_to_output_file.csv> --model_path <final_model>
```

Paramètres :

    --input_file : Le chemin vers votre fichier d'entrée (CSV contenant les abstracts).
    --output_file : Le chemin où vous souhaitez enregistrer le fichier de sortie avec les prédictions.
    --model_path : Le chemin vers le modèle fine-tuné (par défaut "final_model").