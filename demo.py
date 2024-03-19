import tkinter as tk
from tkinter import filedialog, messagebox, Label, PhotoImage
import pandas as pd
import os
import torch 
import clip
import numpy as np
from PIL import Image
import string
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device,"device")

models = clip.available_models()
print(models)
model, preprocess = clip.load('RN50', device,jit=False)

def load_image():
    file_path = filedialog.askopenfilename(filetypes=[("JPG Files", "*.jpg")])
    if file_path:
        messagebox.showinfo("Image Loaded", "Image loaded successfully!")
    return file_path

def method_1():
    messagebox.showinfo("Method 1", "You have chosen Method 1.")
    file_path = load_image()
    if file_path:
        summary_bert_f(file_path)

def method_2():
    messagebox.showinfo("Method 2", "You have chosen Method 2.")
    #main_ingred(file_path)

def method_3():
    messagebox.showinfo("Method 3", "You have chosen Method 3.")
    #all_ingredients(file_path)


# Initialise la fenêtre principale
root = tk.Tk()
root.title("Image Processing App")

# Configuration de la fenêtre pour qu'elle apparaisse au centre de l'écran
window_width = 600
window_height = 400

# Obtenez les dimensions de l'écran
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# Calculez la position x et y pour placer la fenêtre au centre
center_x = int(screen_width/2 - window_width / 2)
center_y = int(screen_height/2 - window_height / 2)

# Définissez la taille de la fenêtre et positionnez-la
root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')

# Ajoutez de la couleur à l'arrière-plan de la fenêtre
root.configure(bg='light blue')

# Ajoutez un texte de bienvenue
welcome_label = Label(root, text="Welcome to our project", bg='light blue', font=('Arial', 20))
welcome_label.pack(pady=20)

# Bouton pour charger une image
load_button = tk.Button(root, text="Load JPG Image", command=load_image, bg='light grey')
load_button.pack(pady=10)

# Boutons pour les trois méthodes
method1_button = tk.Button(root, text="Using bert summaries", command=method_1, bg='light grey')
method1_button.pack(pady=10)

method2_button = tk.Button(root, text="Using GPT4 to choose 2 main ingredients", command=method_2, bg='light grey')
method2_button.pack(pady=10)

method3_button = tk.Button(root, text="Using all ingredients", command=method_3, bg='light grey')
method3_button.pack(pady=10)

# Lancer la boucle principale de la fenêtre
root.mainloop()







def summary_bert_f(file_path):

    summary_bert_path = "/users/eleves-b/2022/hanna.mergui/Computer-Vision/ComputerVision_Data/Summaries/Summary_Bert.csv"
    summary_bert = pd.read_csv(summary_bert_path)
    recipe = summary_bert["Summary"]
    recipe = recipe[:200]

    tensor_bert = torch.load('tensor_bert.pt')
    image = preprocess(Image.open(file_path)).unsqueeze(0).to(device)

    with torch.no_grad():
    
        image_features = model.encode_image(image)
        text_features = model.encode_text(tensor_bert)
    
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)


    values, indices = torch.topk(similarity[0], k=5)

    # Convert the tensors to numpy arrays
    values = values.cpu().numpy()
    indices = indices.cpu().numpy()

    # Create a list of tuples containing the values and indices
    values_indices = list(zip(values, indices))

    # Sort the list in descending order based on the values
    values_indices.sort(reverse=True)

    # Extract the sorted values and indices
    sorted_values, sorted_indices = zip(*values_indices)

    max_index=sorted_indices[0] #text qui correspond à l'image selon le modèle 

    # Create a label widget
    result_label = Label(root, text=recipe[max_index], bg='light blue', font=('Arial', 16))
    result_label.pack(pady=10)
    print("Model Answer",recipe[max_index])
