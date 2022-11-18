# !pipenv install torch
# !pipenv install torchvision
# !pipenv install fastai
# !pipenv install fastcore
# !pipenv install opencv-python
# !pipenv install pillow

import torch
from torchvision import transforms
from fastcore.all import *
from fastai.text.all import *
from fastai.vision.all import *
import streamlit as st
from PIL import Image, ImageOps
import numpy as np

with st.spinner("Model is being loaded.."):
    model = load_learner("nail_model_5classes.pkl")

st.write(
    """
      # Nail classifcation
      ##### This app classifies nail images as one of six classes: normalnail (healthy), naildystrophy, melanonychia, onycholysis, onychomycosis. 
      ##### Upload an image of a nail and the app will tell you which class it predicts.
      ##### The model was trained on around 3000 images using transfer learning on a resnet18 model in fastai/PyTorch.
      ##### Disclaimer: This app is for educational purposes only. It may not be used for medical diagnosis.
    """
)

file = st.file_uploader(
    "Upload the image to be classified \U0001F447", type=["jpg", "png"]
)

st.set_option("deprecation.showfileUploaderEncoding", False)

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    image = image.resize((192, 192))
    st.image(image, use_column_width=True)
    image = PILImage.create(np.array(image.convert("RGB")))
    is_healthy, _, probs = model.predict(image)
    st.write("## The image is classified as:", is_healthy)
    st.write(f"##### (with a probability of: {torch.max(probs):.4f})")

    from os import listdir
    from os.path import isfile, join

    list_imgs = [
        f
        for f in listdir(f"./nail_images/{is_healthy}")
        if isfile(join(f"./nail_images/{is_healthy}", f))
    ]

    st.write("##### Here are 6 other images from the same class...")

    image_list = []
    for i in range(6):
        i = np.random.randint(0, len(list_imgs))
        image_add = Image.open(f"./nail_images/{is_healthy}/{list_imgs[i]}")
        image_list.append(image_add.resize((192, 192)))
    st.image(image_list, use_column_width="auto", width=50)
