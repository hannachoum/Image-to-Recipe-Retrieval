import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox

def load_image():
    file_path = filedialog.askopenfilename(filetypes=[("JPG Files", "*.jpg")])
    if file_path:
        messagebox.showinfo("Image Loaded", "Image loaded successfully!")
    return file_path

def method_1():
    messagebox.showinfo("Method 1", "You have chosen Method 1.")
    # Insérez ici le code spécifique à la méthode 1

def method_2():
    messagebox.showinfo("Method 2", "You have chosen Method 2.")
    # Insérez ici le code spécifique à la méthode 2

def method_3():
    messagebox.showinfo("Method 3", "You have chosen Method 3.")
    # Insérez ici le code spécifique à la méthode 3

# Initialise la fenêtre principale
root = tk.Tk()
root.title("Image Processing App")

# Bouton pour charger une image
load_button = tk.Button(root, text="Load JPG Image", command=load_image)
load_button.pack()

# Boutons pour les trois méthodes
method1_button = tk.Button(root, text="Using bert summaries", command=method_1)
method1_button.pack()

method2_button = tk.Button(root, text="Using GPT4 to choose 2 main ingredients", command=method_2)
method2_button.pack()

method3_button = tk.Button(root, text="Using all ingredients", command=method_3)
method3_button.pack()

# Lancer la boucle principale de la fenêtre
root.mainloop()
