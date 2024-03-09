import pandas as pd
import openai
import csv

# Configurer votre clé API OpenAI
openai.api_key = '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'

# Lire le fichier CSV original
df = pd.read_csv('export.csv')

# Créer un nouveau DataFrame pour les résultats
new_rows = []

number_of_augmented_texts = 3

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
count=0
for _, row in df.iterrows():
    if count < 2:
        count += 1
    else:
        break

    title = row['Title']
    instructions = row['Summary']
    
    # Générer 10 nouvelles instructions pour chaque recette
    for i in range(number_of_augmented_texts):
    
        prompt= "Create for me a new string with the same meaning as this sentance : "+instructions
        augmented_text = generate_augmented_texts(prompt)[0]
        print(augmented_text)
        new_rows.append({'title': title, 'augmented_instructions': augmented_text.replace(",", "")})

# Créer un DataFrame avec les nouvelles lignes
new_df = pd.DataFrame(new_rows)

# Sauvegarder le nouveau DataFrame dans un fichier CSV
new_df.to_csv('Instructions_augmentation.csv', index=False)
