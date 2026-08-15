import warnings
import numpy as np
import streamlit as st
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.utils import img_to_array

warnings.filterwarnings("ignore")

EMBEDDINGS_FILE = "data/jewelry_embeddings.npz"
TOP_K = 25                
SIMILARITY_THRESHOLD = 0.6 

@st.cache_resource
def load_backbone():
    backbone = MobileNetV2(
        weights="imagenet",
        include_top=False,
        pooling="avg",   
    )
    backbone.trainable = False
    return backbone

@st.cache_resource
def load_index():
    data = np.load(EMBEDDINGS_FILE, allow_pickle=True)
    feature_list = data["features"]
    image_paths = data["paths"]

    neighbors = NearestNeighbors(n_neighbors=TOP_K, algorithm="brute", metric="cosine")
    neighbors.fit(feature_list)

    return neighbors, image_paths


def preprocess_img(pil_img):
    pil_img = pil_img.convert("RGB").resize((224, 224))
    img_array = img_to_array(pil_img)
    return preprocess_input(img_array)


def extract_embedding(pil_img, backbone):
    preprocessed_img = preprocess_img(pil_img)
    img = np.expand_dims(preprocessed_img, axis=0)
    embedding = backbone.predict(img, verbose=0)
    return embedding  # shape: (1, 1280)

st.set_page_config(page_title="Jewelry Visual Search", layout="wide")
st.title("💍 Jewelry Visual Search Engine")
st.write("Upload a jewelry photo (or use your camera) to find visually similar items in the catalog.")

backbone = load_backbone()
neighbors, image_paths = load_index()

source = st.radio("Choose image source:", ["Upload a photo", "Use camera"], horizontal=True)

if source == "Upload a photo":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
else:
    uploaded_file = st.camera_input("Take a photo")

if uploaded_file is not None:
    query_img = Image.open(uploaded_file)

    st.subheader("Query Image")
    st.image(query_img, width=250)

    with st.spinner("Searching for similar items..."):
        query_embedding = extract_embedding(query_img, backbone)
        distances, indices = neighbors.kneighbors(query_embedding)

    similarities = 1 - distances[0]
    indices = indices[0]

   
    keep = similarities >= SIMILARITY_THRESHOLD
    similarities = similarities[keep]
    indices = indices[keep]

    if len(indices) == 0:
        st.warning("No similar jewelry items found in the catalog for this image.")
    else:
        st.subheader(f"Top {len(indices)} Matches")
        cols = st.columns(5)
        for i, (idx, sim) in enumerate(zip(indices, similarities)):
            with cols[i % 5]:
                st.image(str(image_paths[idx]), use_container_width=True)
                st.caption(f"Similarity: {sim:.3f}")
