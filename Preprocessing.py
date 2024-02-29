import os
import pandas as pd
from PIL import Image
import numpy as np

# Define the paths to your dataset
csv_file_path = 'path/to/your/csv_file.csv'  # Update this path
images_folder_path = 'path/to/your/images_folder'  # Update this path

# Load the dataset
df = pd.read_csv(csv_file_path)

# Function to clean the Ingredients column
def clean_ingredients(ingredients):
    # Implement your cleaning logic here
    cleaned = ingredients.lower()  # Example: convert to lowercase
    # Add more cleaning steps as needed
    return cleaned

# Apply cleaning function to the Ingredients column
df['Cleaned_Ingredients'] = df['Ingredients'].apply(clean_ingredients)

# Function to preprocess images
def preprocess_image(image_name):
    image_path = os.path.join(images_folder_path, image_name)
    try:
        with Image.open(image_path) as img:
            # Example preprocessing: resize and convert to grayscale
            img = img.resize((128, 128)).convert('L')
            img_array = np.array(img)
            return img_array
    except IOError:
        return None

# Example of how to preprocess and save images
for image_name in df['Image_Name']:
    preprocessed_img = preprocess_image(image_name)
    if preprocessed_img is not None:
        # Save or process your images here
        # Example: save the preprocessed image
        save_path = os.path.join('path/to/save/preprocessed_images', image_name)
        Image.fromarray(preprocessed_img).save(save_path)

# Save the cleaned CSV file
df.to_csv('path/to/save/your_cleaned_file.csv', index=False)
