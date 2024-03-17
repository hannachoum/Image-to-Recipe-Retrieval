"""
import pandas as pd
import openai
import csv
import tqdm
from tqdm import tqdm

# Configurer votre clé API OpenAI
openai.api_key = 'cle'

# Lire le fichier CSV original
df = pd.read_csv('ComputerVision_Data/Summaries/export_summary_Bert.csv')

# Créer un nouveau DataFrame pour les résultats
new_rows = []

number_of_augmented_texts = 5

# Fonction pour générer des textes augmentés

def generate_augmented_texts(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # or another model of your choice
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=77,  # Adjust according to the length of the instructions
    )
    return [response['choices'][0]['message']['content']]

# Parcourir chaque ligne du DataFrame

with open('ComputerVision_Data/Summaries/Text_augmentation.csv', 'a', newline='') as csvfile:
    writer = csv.writer(csvfile)
    
    for _, row in tqdm(df.iterrows()):
        title = row['Title']
        instructions = row['Summary']
        
        # Générer 10 nouvelles instructions pour chaque recette
        for i in range(number_of_augmented_texts):
            prompt= "Create for me a new string with the same meaning as this sentence: " + instructions + ". Give me directly the new string."
            augmented_text = generate_augmented_texts(prompt)[0]
            new_row = [title, augmented_text.replace(",", "")]
            writer.writerow(new_row)
"""


import pandas as pd
import openai
import csv
from tqdm import tqdm

# Configurer votre clé API OpenAI
openai.api_key = 'cle' 

# Lire le fichier CSV original
df = pd.read_csv('ComputerVision_Data/Summaries/export_summary_Bert.csv')

# La variable commence_après définie le titre à partir duquel commencer le processus
commence_apres = "Buttery Pull-Apart Dinner Rolls"
commencer = False

# Créer un nouveau DataFrame pour les résultats
new_rows = []

number_of_augmented_texts = 5

# Fonction pour générer des textes augmentés
def generate_augmented_texts(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # or another model of your choice
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=77,  # Adjust according to the length of the instructions
    )
    return [response['choices'][0]['message']['content']]

# Parcourir chaque ligne du DataFrame
with open('ComputerVision_Data/Summaries/Text_augmentation.csv', 'a', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        title = row['Title']
        
        # Vérifie si le processus doit commencer ou non
        if not commencer and title == commence_apres:
            commencer = True
        if not commencer:
            continue

        instructions = row['Summary']
        
        # Générer des nouvelles instructions pour chaque résumé
        for i in range(number_of_augmented_texts):
            prompt= "Create for me a new string with the same meaning as this sentence: " + instructions + ". Give me directly the new string."
            augmented_text = generate_augmented_texts(prompt)[0]
            new_row = [title, augmented_text.replace(",", "")]
            writer.writerow(new_row)

