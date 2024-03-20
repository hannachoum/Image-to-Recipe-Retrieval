from flask import Flask, request, redirect, url_for, render_template_string
import os
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
data_dir = "data"

models = clip.available_models()
print(models)
model, preprocess = clip.load('ViT-B/32', device,jit=False)



app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

TEMPLATE = '''
<!doctype html>
<html>
<head>
    <title>Upload and Choose</title>
    <style>
        body {
            background: url('https://assets-global.website-files.com/621384900cbdd71138c16c99/65d4baa987c97ad917011d4f_Cover%20blog-1-min.jpg') no-repeat center center fixed;
            background-size: cover;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            font-family: Arial, sans-serif;
        }
        .container {
            background-color: rgba(255, 255, 255, 0.8);
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            text-align: center;
        }
        input[type="file"] {
            margin: 20px 0;
        }
        input[type="submit"] {
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
        input[type="submit"]:hover {
            background-color: #45a049;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Give me a food image and I will try to give you the recipe ! </h1>
        <form method="post" enctype="multipart/form-data">
            <input type="file" name="file"><br>
            <input type="radio" name="choice" value="option1" checked> Summary with Bert <br>
            <input type="radio" name="choice" value="option2"> 2 main ingredients with GPT 4 2<br>
            <input type="radio" name="choice" value="option3"> All ingredients 3<br>
            <br>
            <input type="submit" value="Upload and Choose">
        </form>
        <br>
        
        {{ result_message }}
    </div>
</body>
</html>
'''

print('before')

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    result_message = ""
    if request.method == 'POST':
        
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        
        choice = request.form.get('choice')

        print("File part in request")
        print("Choice selected:", choice)

        filepath="data/Food Images/Food Images/-cod-with-mussels-chorizo-fried-croutons-and-saffron-mayonnaise-355204.jpg"
        
        file.save(filepath)
        if choice == 'option1':
            #result_message = "You chose option 1."
            print('hello')
            result_message = models(filepath,"data/tensors/bert_tensor.pt")
        elif choice == 'option2':
            result_message = models(filepath,data_dir+"/tensors/ingredients_tensor.pt")
        elif choice == 'option3':
            result_message = models(filepath,data_dir+"/tensors/All_ingredients_tensor.pt")
    return render_template_string(TEMPLATE, result_message=result_message)



mapping_path = data_dir+"/NewMapping.csv"
mapping_csv = pd.read_csv(mapping_path)


def models(file_path,tensor_path):

    print('hello')
    tensor_bert = torch.load(tensor_path)
    image = preprocess(Image.open(file_path)).unsqueeze(0).to(device)

   
    with torch.no_grad():
    
        image_features = model.encode_image(image)
        text_features = model.encode_text(tensor_bert)
    
    print("after no grad")
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)


    values, indices = torch.topk(similarity[0], k=5)

    values = values.cpu().numpy()
    indices = indices.cpu().numpy()
    values_indices = list(zip(values, indices))
    values_indices.sort(reverse=True)
    sorted_values, sorted_indices = zip(*values_indices)

    max_index=sorted_indices[0]  

    #print(recipe[max_index])
    #print(max_index)
    print (mapping_csv["Instructions"][max_index])
    return mapping_csv["Instructions"][max_index]



if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True, host='0.0.0.0')






