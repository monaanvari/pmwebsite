import streamlit as st
import os
from PIL import Image
import generate
from torch.autograd import Variable
import torch.nn as nn
from torchvision.models import resnet
import argparse


parser = argparse.ArgumentParser(description='Template-based parametric material prediction')
args = parser.parse_args()
args.mlp = 1
args.base_model = 'resnet34'
args.num_classes=314
args.regression=1
args.alpha=1e-2

st.title("Photorealistic Materials")


img = st.sidebar.file_uploader("Choose an image...", type="jpeg")
if img is not None:
    ii = (Image.open(img).save(
        "images/content-images" + "content.jpeg"))
    input_image = "images/content-images" + "content.jpeg"

# TODO: delete content image after done



model = "saved_models/model_50.pth.tar"

if img is not None:
    st.write("### Source Image:")
  
    image = Image.open(img)

    st.image(image, caption='Uploaded Image.', width=300)

clicked = st.button("Generate Material")
if clicked:
    model = generate.load_pm_model(model,args)
    rendering_path = generate.generate_material(model, image) #get image and return rendering 



    st.write("### Output Image:")

    image = Image.open(rendering_path)

    st.image(image, caption='Output Image.', width=300)
