import streamlit as st
import base64
import os
from io import BytesIO
from PIL import Image
import pandas as pd
import generate
import torch.nn as nn
from torchvision.models import resnet
from zipfile import ZipFile
import parse_args
from google_drive_downloader import GoogleDriveDownloader as gdd

samplepath = "model_output/predictions/mymat.txt"
mdlpath = "/cvgl2/u/monaavr/nvidia_mdl_templates/nvidia/vMaterials/predictions"
imagespath = "images/content-images"
modelpath = "saved_models/model_50.pth.tar"

def main():
    args = parse_args.parse_args()
    st.title("Generate Photorealistic Materials")

    img = st.sidebar.file_uploader("Choose an image...", type="jpeg")
    if img is not None:
        ii = (Image.open(img).save(
            imagespath + "content.jpeg"))
        input_image = imagespath + "content.jpeg"

    gdd.download_file_from_google_drive(file_id='1OHQmKb_d_xLrkdzYlo8_9Vf7F6IFIiiE',
                                        dest_path=modelpath,
                                        unzip=False)

    if img is not None:
        st.write("### Source Image:")
        image = Image.open(img)
        st.image(image, caption='Uploaded Image.', width=420)

    clicked = st.button("Generate Material")
    if clicked:
        model = generate.load_pm_model(modelpath,args)
        rendering_path = generate.generate_material(model, image) #get image and return rendering 
        st.write("### Output Image:")
        image = Image.open(rendering_path)
        st.image(image, caption='Output Image.', width=420)
        tmp_link = download_link(mdlpath)
        st.markdown(tmp_link, unsafe_allow_html=True)


def download_link(mdlpath):

    zipfilepath= "mymat.zip"
    zf = ZipFile(zipfilepath, "w")
    for dirname, subdirs, files in os.walk(mdlpath):
        zf.write(dirname)
        for filename in files:
            zf.write(os.path.join(dirname, filename))
    zf.close()

    with open(zipfilepath, "rb") as f:
        bytes = f.read()
        b64 = base64.b64encode(bytes).decode()
        href = f"<a href=\"data:file/zip;base64,{b64}\" download='{zipfilepath}.zip'> Download material as MDL</a>"
        return href


if __name__ == "__main__":
    main()