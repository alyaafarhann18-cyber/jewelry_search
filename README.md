
# 💍 Jewelry Visual Search Engine
 
A production-style **visual similarity search engine** for jewelry images. Upload a photo of a ring or necklace (or take one with your camera), and the app retrieves the most visually similar items from the catalog using deep learning embeddings.
 
Built as part of the *Building a Production-Grade Visual Search Engine* exercise (General Assembly Data Science program, in partnership with BIBF).
 
---
 
## How It Works
 
1. **Feature Extraction** — Every catalog image is passed through **MobileNetV2** (pretrained on ImageNet, classification head removed) to produce a 1280-dimensional embedding vector that captures its visual characteristics.
2. **Offline Indexing** — Embeddings are pre-computed once and saved to disk, so the app never has to re-process the entire catalog on every search.
3. **Similarity Search** — When a user uploads a query image, its embedding is extracted the same way, and compared against the catalog using **cosine similarity** via scikit-learn's `NearestNeighbors`.
4. **Results** — The top 25 most similar items (above a similarity threshold) are displayed to the user.
---
 
## Project Structure
 
```
jewelry_search/
├── data/
│   ├── Jewellery_Data/          # raw images (ring/, necklace/)
│   └── jewelry_embeddings.npz   # pre-computed embeddings + paths
├── prepare_data.py              # offline pipeline: builds the embeddings index
├── app.py                       # Streamlit web application
└── requirements.txt
```
 
---
 
## Dataset
 
Jewelry images (rings and necklaces), organized into subfolders:
 
```
Jewellery_Data/
├── ring/
└── necklace/
```
 
---
 
## Tech Stack
 
- **TensorFlow / Keras** — MobileNetV2 backbone for embedding extraction
- **scikit-learn** — `NearestNeighbors` for cosine-similarity search
- **Streamlit** — interactive web interface
- **NumPy / Pillow** — data handling and image processing
---
 
## Setup
 
```bash
git clone https://github.com/alyaafarhann18-cyber/jewelry_search.git
cd jewelry_search
pip install -r requirements.txt
```
 
Place the dataset under `Jewellery_Data/` at the project root (with `ring/` and `necklace/` subfolders).
 
---
 
## Usage
 
**1. Build the embeddings index (run once, or whenever the catalog changes):**
 
```bash
python prepare_data.py
```
 
This scans `Jewellery_Data/`, extracts an embedding for every image, and saves them to `data/jewelry_embeddings.npz`.
 
**2. Launch the app:**
 
```bash
https://jewelrysearch-ayeqy6wf7bypenubr7yu48.streamlit.app/
```
 
## Notes
 
- The similarity threshold (`SIMILARITY_THRESHOLD` in `app.py`) controls how strict matching is — queries with no good match (e.g. an unrelated object) will return no results instead of irrelevant ones.
- Model loading and the search index are cached with `@st.cache_resource` so the app stays fast after the first load.
