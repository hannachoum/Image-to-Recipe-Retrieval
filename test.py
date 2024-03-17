import pandas as pd
import os
import torch 
import clip
import numpy as np
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"


summary_bert_path = "/users/eleves-b/2022/hanna.mergui/Computer-Vision/ComputerVision_Data/Summaries/export_summary_Bert.csv"
summary_bert = pd.read_csv(summary_bert_path)

data_dir = "ComputerVision_Data"
images_dir = "Food Images/Food Images"


recipe = summary_bert["Summary"]


images = os.listdir(os.path.join(data_dir, images_dir))


models = clip.available_models()
model, preprocess = clip.load('RN50', device,jit=False)



#Tensor creation and array creation


context_length=77
array_text=[]


#print(device,"device")
text_tensor = torch.zeros(len(summary_bert), context_length, dtype=torch.long)
labels = []
for i, row in enumerate(summary_bert.iterrows()):
    summary = row[1]["Summary"]
    if len(summary)>context_length:
        summary = summary[:context_length]
    array_text.append(summary)
    text_tensor[i] = clip.tokenize(summary, context_length).to(device)  # Access row data using integer indices
    labels.append(row[1]["Image_Name"])  # Access row data using integer indices

torch.save(text_tensor, "text_tensor.pt")

text_tensor = torch.load("text_tensor.pt")

images_inputs=[]

for im in images:
    im_path = os.path.join(data_dir, images_dir, im)
    images_inputs.append(preprocess(Image.open(im_path)))
images_inputs_tensor = torch.tensor(np.stack(images_inputs)).to(device)



image = preprocess(Image.open("ComputerVision_Data/Food Images/Food Images/-bloody-mary-tomato-toast-with-celery-and-horseradish-56389813.jpg")).unsqueeze(0).to(device)


text = clip.tokenize(array_text, context_length).to(device)


with torch.no_grad():
    #print("image is",image)
    #print("text is",text)
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

#print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]
print("best prob",max(probs[0]))
best_index = np.argmax(probs[0])
print("best index:", best_index)