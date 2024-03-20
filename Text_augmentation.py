"""
import pandas as pd
import openai
import csv
import tqdm
from tqdm import tqdm

# Configurer votre clé API OpenAI

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
# set openai.api_key

# Lire le fichier CSV original
df = pd.read_csv('data/Summaries/Summary_Bert_77.csv')

# Créer un nouveau DataFrame pour les résultats
new_rows = []

number_of_augmented_texts = 2

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
with open('data/Summaries/Text_augmentation.csv', 'a', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Title', 'Ingredients', 'Instructions', 'Image_Name', 'Cleaned_Ingredients', "Summary_Bert_77", "Text_augmented_1", "Text_augmented_2"])
    
    for y, row in tqdm(df.iterrows(), total=df.shape[0]):
        title = row['Title']
        Ingredients = row ['Ingredients']
        Instructions = row ['Instructions']
        Image_Name = row ['Image_Name']
        Cleaned_Ingredients = row ['Cleaned_Ingredients']
        Summary_Bert_77 = row ['Summary_Bert_77']
        
        # Générer des nouvelles instructions pour chaque résumé
        for i in range(number_of_augmented_texts):
            prompt= "Create for me a new string with the same meaning as this recipe summary : " + Summary_Bert_77 + ". Give me directly the new string."
            if i==0:
                augmented_text = generate_augmented_texts(prompt)[0]
                new_row1 = augmented_text.strip().replace(",", "")
            elif i==1:
                augmented_text2 = generate_augmented_texts(prompt)[0]
                new_row2 = augmented_text2.strip().replace(",", "")
        

        if y==0:
            print([title, Ingredients, Instructions, Image_Name, Cleaned_Ingredients, Summary_Bert_77, new_row1, new_row2])

        writer.writerow([title, Ingredients, Instructions, Image_Name, Cleaned_Ingredients, Summary_Bert_77, new_row1, new_row2])
