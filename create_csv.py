import pandas as pd
from transformers import pipeline
from tqdm import tqdm
import torch
import csv

# Charger le fichier CSV
input_path = pd.read_csv('data/Food Ingredients and Recipe Dataset with Image Name Mapping.csv')
output_csv_path = 'bert_result_multi.csv'
MAX_LENGTH = 77

# Configuration du dispositif pour l'exécution du modèle
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialiser le modèle de résumé
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0 if device == "cuda" else -1)
nb_toolong = 0

count=0

with open(output_csv_path, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    # Écrire l'en-tête du fichier CSV
    writer.writerow(['Title', 'Ingredients', 'Summary', 'Image_Name', 'Cleaned_Ingredients'])

    for _, row in tqdm(input_path.iterrows(), desc="Processing descriptions"):
        description = row['Instructions']
        title = row.get('Title', '')  # Remplacer 'Title' par le nom réel de votre colonne
        ingredients = row.get('Ingredients', '')  # Idem
        image_name = row.get('Image_Name', '')  # Idem
        cleaned_ingredients = row.get('Cleaned_Ingredients', '')  # Idem

        count+=1
        
        if count == 100:
            break
        
        else: 
            # Générer le résumé
            if len(description) < 4000:
                if len(description) < MAX_LENGTH:
                    summary = description
                else:
                    summary_result = summarizer(description, max_length=MAX_LENGTH, min_length=10, do_sample=False)
                    #print("1")
                    #print (summary_result)
                    summary = summary_result[0]['summary_text']
                    #print(summary)
            elif len(description)/2 < 4000:
                first_part = description[:int(len(description) / 2)]
                second_part = description[int(len(description) / 2):int(len(description))]


                first_summary = summarizer(first_part, max_length=MAX_LENGTH, min_length=10, do_sample=False)
                #print("first 1",first_summary)
                first_summary = first_summary[0]['summary_text']
                #print("first 2",first_summary)

                second_summary = summarizer(second_part, max_length=MAX_LENGTH, min_length=10, do_sample=False)
                #print(second_summary)
                second_summary = second_summary[0]['summary_text']
                #print(second_summary)
                summary = first_summary + " " + second_summary
                #print(second_summary)
            else:
                nb_toolong += 1
                summary = description

            # Écrire la ligne dans le fichier CSV
            writer.writerow([title, ingredients, summary, image_name, cleaned_ingredients])

#print(f"{nb_toolong} descriptions were too long for summarization.")
