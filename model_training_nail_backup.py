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

# Path where the images are saved
path = Path('nail_images')

# Set up data block
dls = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=[Resize(192, method='squish')] 
).dataloaders(path, bs=32)

# Set up model and fine tune it 
learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(10)

# Prediction using any image
is_healthy,_,probs = learn.predict(PILImage.create('nail_predict.jpg'))
print(f"This is a: {is_healthy}.")
print(f"Probability it's a healthy nail: {probs[0]:.4f}")

# Save model
learn.export("nail_model.pkl")

# Load model
nail_model = load_learner("nail_model.pkl")

# Prediction using any image
is_healthy,_,probs = nail_model.predict(PILImage.create('nail_predict.jpg'))
print(f"This is a: {is_healthy}.")
print(f"Probability it's a healthy nail: {probs[0]:.4f}")

# image = Image.open('nail_predict.jpg')
# image = image.resize((192,192))
# st.image(image, use_column_width=True)
# image = PILImage.create(np.array(image.convert('RGB')))
        
# is_healthy,_,probs = nail_model.predict(image)
# print(f"This is a: {is_healthy}.")
# print(f"Probability it's a healthy nail: {probs[0]:.4f}")

with st.spinner('Model is being loaded..'):
  model=load_learner("nail_model.pkl")

st.write("""
         # Image Classification
         """
         )

file = st.file_uploader("Upload the image to be classified \U0001F447", type=["jpg", "png"])
import cv2
from PIL import Image, ImageOps
import numpy as np
st.set_option('deprecation.showfileUploaderEncoding', False)

# def upload_predict(upload_image, model):
    
#         size = (180,180)    
#         image = ImageOps.fit(upload_image, size, Image.ANTIALIAS)
#         image = np.asarray(image)
#         img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         img_resize = cv2.resize(img, dsize=(224, 224),interpolation=cv2.INTER_CUBIC)
        
#         img_reshape = img_resize[np.newaxis,...]
    
#         prediction = model.predict(img_reshape)
#         pred_class=decode_predictions(prediction,top=1)
        
#         return pred_class

if file is None:
    st.text("Please upload an image file")
else:
    
    image = Image.open('nail_predict.jpg')
    image = image.resize((192,192))
    st.image(image, use_column_width=True)
    image = PILImage.create(np.array(image.convert('RGB')))
    is_healthy,_,probs = nail_model.predict(image)
    st.write("The image is classified as", is_healthy)
    st.write("With a probability of", probs)
    print("The image is classified as ",is_healthy, "with a similarity score of",probs)