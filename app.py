import streamlit as st
from PIL import Image
import numpy as np
import tempfile
import os

from script import process_image

st.set_page_config(layout="wide")
st.title("Laws Texture + K-Means Segmentation")

uploaded_file = st.file_uploader("Upload a grayscale satellite image", type=["png", "tif", "tiff"])

k = st.slider("Number of texture clusters (k)", 2, 10, 4)

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    segmented = process_image(tmp_path, k)
    segmented_img = Image.fromarray(segmented)

    original_img = Image.open(tmp_path).convert("L").resize(segmented_img.size)

    col1, col2 = st.columns(2)
    with col1:
        st.image(original_img, caption="Original Image", use_column_width=True)
    with col2:
        st.image(segmented_img, caption=f"Segmented Image (k={k})", use_column_width=True)

    os.remove(tmp_path)
