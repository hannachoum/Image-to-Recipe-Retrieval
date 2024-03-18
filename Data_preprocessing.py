import clip
import pandas as pd
import os
import torch 
import numpy as np
import string
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertForSequenceClassification
import json
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import clip
from transformers import CLIPProcessor, CLIPModel
import tqdm
from transformers import CLIPTextConfig, CLIPTextModel
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm 
import mlflow

data_dir = "/users/eleves-b/2022/hanna.mergui/Computer-Vision/ComputerVision_Data"
images_dir = "/users/eleves-b/2022/hanna.mergui/Computer-Vision/ComputerVision_Data/Mapping.csv"

origin_data = pd.read_csv('/users/eleves-b/2022/hanna.mergui/Computer-Vision/ComputerVision_Data/Mapping.csv')
print(len(origin_data)) #13463

# Remove rows with no image name
origin_data = origin_data[origin_data['Image_Name']!='#NAME?']
print(len(origin_data)) #13463


recipe = origin_data["Instructions"].tolist()
print("number of Instructions recipe ",len(recipe)) #13463

Instructions=origin_data["Instructions"].tolist()
Ingredients=origin_data["Ingredients"].tolist()

to_supress = string.punctuation + "—" + "–" + "()"

for i in range(len(Ingredients)): 
    temp = Ingredients[i].translate(str.maketrans("", "",to_supress))
    temp= ''.join([char for char in temp if not char.isdigit()])
    Ingredients[i]=temp.replace("Tbsp", "").replace("½", "").replace("¾", "").replace("lb", "").replace("tsp", "").replace("⅓", "").replace("  ", "").replace("¼", "")
  
Instructions = [instruction.translate(str.maketrans("", "", to_supress)).replace("Tbsp", "").replace("½", "").replace("¾", "").replace("lb", "").replace("tsp", "").replace("⅓", "").replace("  ", "").replace("¼", "") for instruction in Instructions]

# Create a new DataFrame with updated columns
new_data = pd.DataFrame({'Title': origin_data['Title'], 'Ingredients': Ingredients, 'Instructions': Instructions, 'Image_Name': origin_data['Image_Name'], 'Cleaned_Ingredients': origin_data['Cleaned_Ingredients']})

# Save the new DataFrame to a CSV file
new_data.to_csv('/users/eleves-b/2022/hanna.mergui/Computer-Vision/ComputerVision_Data/NewMapping.csv', index=False)
# Save the new DataFrame to a CSV file


