import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import csv
import torch
from tqdm import tqdm

# Assuming GemmaForCausalLM is a correct class name, make sure it's available in your transformers package
# from transformers import GemmaForCausalLM

max_length = 77

# Configuration for model execution
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("google/gemma-7b").to(device)
tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b")

# Open the input CSV file
with open('ComputerVision_Data/Food Ingredients and Recipe Dataset with Image Name Mapping.csv', 'r') as csvfile:
    reader = csv.DictReader(csvfile)

    # Open the output CSV file
    with open('export_summary_gemma.csv', 'w', newline='') as outfile:
        writer = csv.writer(outfile)

        # Write the header row in the output CSV file
        writer.writerow(['Title', 'Ingredients', 'Image_Name', 'Cleaned_Ingredients', 'Summary'])

        count = 0

        # Loop over each row in the input CSV file
        for row in tqdm(reader):
            if count > 2:
                break

            Title = row['Title']
            Ingredients = row['Ingredients']
            Image_Name = row['Image_Name']
            Instructions = row['Instructions']
            Cleaned_Ingredients = row['Cleaned_Ingredients']

            # Generate summary using Gemma
            prompt = "Make a summary of: " + Instructions
            input_ids = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)

            outputs = model.generate(**input_ids, max_new_tokens=max_length, num_beams=5, no_repeat_ngram_size=2)

            summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
            summary = summary.replace(",", "")

            # Write the description and summary to the output CSV file
            writer.writerow([Title, Ingredients, Image_Name, Cleaned_Ingredients, summary])

            count += 1
