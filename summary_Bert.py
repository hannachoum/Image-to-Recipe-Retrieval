import json
import pandas as pd
from transformers import pipeline
from tqdm import tqdm
import torch
import csv

# Charger le fichier CSV
input_path = pd.read_csv('ComputerVision_Data/Food Ingredients and Recipe Dataset with Image Name Mapping.csv')
output_csv_path = 'export_summary_Bert.csv'
MAX_LENGTH = 77

# Configuration du dispositif pour l'exécution du modèle
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialiser le modèle de résumé
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0 if device == "cuda" else -1)
nb_toolong = 0
idx = 0

# Remove rows with no image name
input_path = input_path[input_path['Image_Name']!='#NAME?']


def count_tokens(text):
    # Remplacer les signes de ponctuation par des espaces et diviser le texte en mots
    words = text.replace(',', ' ').replace('.', ' ').replace('!', ' ').replace('?', ' ').split()
    return len(words)


with open(output_csv_path, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    # Écrire l'en-tête du fichier CSV
    writer.writerow(['Title', 'Ingredients', 'Summary', 'Image_Name', 'Cleaned_Ingredients'])
    not_str = {}

    for _, row in tqdm(input_path.iterrows(), desc="Processing descriptions"):
        description = row['Instructions']
        title = row.get('Title', '')  # Remplacer 'Title' par le nom réel de votre colonne
        ingredients = row.get('Ingredients', '')  # Idem
        image_name = row.get('Image_Name', '')  # Idem
        cleaned_ingredients = row.get('Cleaned_Ingredients', '')  # Idem

        # print(len(description), type(description))
        # print(description)
        # Générer le résumé
        if not isinstance(description, str) or len(str(description))==0:
            not_str[idx] = [image_name, description]
            idx+=1
            continue
        if len(description) < 4000:
            if len(description.split()) < MAX_LENGTH:
                summary = description
            else:
                summary_result = summarizer(description, max_length=MAX_LENGTH, min_length=40, do_sample=False)
                # print ("summary_res", summary_result)
                summary = summary_result[0]['summary_text']
                # print("summary", summary)

        # case where input too long to smmerize
        elif len(description)/2 < 4000:
            first_part = description[:int(len(description) / 2)]
            second_part = description[int(len(description) / 2):int(len(description))]


            first_summary = summarizer(first_part, max_length=MAX_LENGTH, min_length=10, do_sample=False)
            #print("first 1",first_summary)
            first_summary = str(first_summary[0]['summary_text'])
            #print("first 2",first_summary)

            second_summary = summarizer(second_part, max_length=MAX_LENGTH, min_length=10, do_sample=False)
            #print(second_summary)
            second_summary = str(second_summary[0]['summary_text'])
            #print(second_summary)
            summary = str(first_summary + " " + second_summary)
            #print(second_summary)
            summary = summarizer(summary, max_length=MAX_LENGTH, min_length=10, do_sample=False)
            summary = str(summary[0]['summary_text'])

        else:
            nb_toolong += 1
            summary = "too long to summarize"

        # Écrire la ligne dans le fichier CSV
        idx+=1
        number_of_tokens = count_tokens(summary)
        if number_of_tokens>MAX_LENGTH:
            print(f"Summary" + title +" too long: {number_of_tokens} tokens")
       
        writer.writerow([title, ingredients, summary, image_name, cleaned_ingredients])

print(f"{nb_toolong} descriptions were too long for summarization.")

with open("bad_instructions.txt", "w") as f:
    json.dump(not_str, f)
