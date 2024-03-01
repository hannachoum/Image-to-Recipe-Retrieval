import pandas as pd
import openai
from transformers import pipeline
import torch
from tqdm import tqdm



# Charger le fichier CSV
df = pd.read_csv('data_recipe/Food Ingredients and Recipe Dataset with Image Name Mapping.csv')

# Assurez-vous de remplacer 'Instructions' par le nom réel de votre colonne contenant les descriptions
colonne_descriptions = df['Instructions']

device = "cuda" if torch.cuda.is_available() else "cpu"

summarize_description=[]
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
nb_toolong=0


for description in tqdm(colonne_descriptions, desc="Processing descriptions"):
    assert isinstance(description, str) 
    assert len(description)!=0

    max = 77
    min =10

    #print(len(description))
    
    if len(description) < 4000:
        if len(description) < max:
            #print("1",len(description))
            summarize_description.append(description)
            #print("1")
        else:
            #print("2",len(description))
            summary=summarizer(description, max_length=max, min_length=min, do_sample=False)
            summarize_description.append(summary)
            #print("2")
            #print("summary",summary)


    elif len(description)/2 <4000: 
        #print("split to summarize")
        #summarize_description.append(description)
        first_part = description[0:int(len(description)/2)]
        second_part = description[int(len(description)/2):]
        #print("3",len(first_part),len(second_part))
        
        first_summary = summarizer(first_part, max_length=max, min_length=min, do_sample=False)[0]['summary_text']
        second_summary = summarizer(second_part, max_length=max, min_length=min, do_sample=False)[0]['summary_text']

        # Concaténez les deux résumés pour obtenir un seul texte de résumé
        total_summary = first_summary + " " + second_summary
        
        summarize_description.append(total_summary)
        #print("3")
        #print(description)
        #print("summary",total_summary)

    else: 
        #print("too long to summarize")
        nb_toolong+=1
        summarize_description.append(description)
        #print("4")

    


texts_only = [item[0]['summary_text'] for item in summarize_description if isinstance(item, list) and len(item) > 0]


df = pd.DataFrame(texts_only, columns=['Summary'])
df.to_csv('bert_result.csv', index=False)


print(nb_toolong,"sentences too long for bert")

