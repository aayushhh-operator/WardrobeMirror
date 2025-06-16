# 👗 Fashion Product Similarity Web App

A Flask-based web application that allows users to upload an image of a fashion product, detects and crops the relevant product using Roboflow API, and returns the top 10 visually similar products from a precomputed dataset using ResNet50 and cosine similarity.

---

## ✨ Features

- 📸 Upload product images for visual similarity search
- 🧠 Uses pretrained ResNet50 for feature extraction
- 🧮 Calculates cosine similarity to find top-10 similar items
- 📦 Bounding box detection via Roboflow hosted model
- 🌐 User-friendly and responsive web UI
- 📊 Displays product names and images

---

To run the app, first type 

pip install -r requirements.txt

in the terminal

then type

python app.py

and then click on the local host url to access the web page.

--- 
## MY APPROACH

The first thing I did was, I cleaned the dataset. The entire code for the cleaning is given in "notebooks\cleaning (1).ipynb".

Next thing that I did was that, I split the entire dataset into multiple datasets consisting of 2000 rows each, and then for each small dataset, I used a roboflow bounding box model to detect on the clothing part of the image to have a better model. All the working related to the bounding box is given in "notebooks\bounding_box_demo (1).ipynb" and "notebooks\split_bounding_box.ipynb".

Once I had the bounding box coordinates, I made a CNN model to extract the features of the cropped image inside the bounding box for each image in the dataset and then stored these features in the "models\similarity_model.pkl".

Once this was done, I made a simple flask app, which takes the input image from the user, crops it with the same roboflow bounding box model and then extracts the features of the cropped input image. It then compares the input image features with the features of the images in the dataset and by cosine similarity, it displays the top 10 matches.

---

Made By Aayush🤖
