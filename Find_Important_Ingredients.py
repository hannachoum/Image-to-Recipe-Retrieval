import pandas as pd
import json
from transformers import pipeline
from tqdm import tqdm
import torch
import csv
import openai

# Read the CSV file

input_path = pd.read_csv('ComputerVision_Data/Summaries/Summary_Bert.csv')
output_csv_path = 'ComputerVision_Data/Summaries/most_important_ingredients.csv'

# set openai.api_key

def generate_augmented_texts(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # or another model of your choice
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=15,  # Adjust according to the length of the instructions
    )
    return [response['choices'][0]['message']['content']]


def count_tokens(text):
    # Remplacer les signes de ponctuation par des espaces et diviser le texte en mots
    words = text.replace(',', ' ').replace('.', ' ').replace('!', ' ').replace('?', ' ').split()
    return len(words)



with open(output_csv_path, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    # Écrire l'en-tête du fichier CSV
    writer.writerow(['Title', 'Ingredients', 'summary_with_ingredients', 'Image_Name', 'Cleaned_Ingredients'])
    not_str = {}

    count=0

    for i, row in tqdm(input_path.iterrows(), desc="Processing descriptions"):
            if i<220:
                title = row.get('Title', '') 
                ingredients = row.get('Ingredients', '')
                summary = row.get('Summary')
                image_name = row.get('Image_Name', '')  # Idem
                cleaned_ingredients = row.get('Cleaned_Ingredients', '')  # Idem

                # Find the two most important words in the ingredients column

                prompt= "I want you to give me the two main ingredients in this list " + ingredients + ". Give me just two words."
                top_ingredients = generate_augmented_texts(prompt)[0]
                #print(" top ingredients",top_ingredients)

                swi = top_ingredients
                swi.replace(",", " ")

                #print("summary with ingredients",swi)
                new_row = []

                if count_tokens(swi)<=77: 
                    writer.writerow([title,ingredients,swi,image_name,cleaned_ingredients])
                else:
                    writer.writerow([title,ingredients,summary,image_name,cleaned_ingredients]) 

