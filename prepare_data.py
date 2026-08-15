import glob
import warnings
import numpy as np
from tqdm import tqdm
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.utils import load_img, img_to_array

warnings.filterwarnings("ignore")


DATA_DIR = "Jewellery_Data/"   # contains: ring/, necklace/
OUTPUT_FILE = "data/jewelry_embeddings.npz"


backbone = MobileNetV2(
    weights="imagenet",
    include_top=False,
    pooling="avg",   
)
backbone.trainable = False


def preprocess_img(img_path):
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img)
    return preprocess_input(img_array)


def extract_embedding(img_path):
    preprocessed_img = preprocess_img(img_path)
    img = np.expand_dims(preprocessed_img, axis=0)
    embedding = backbone.predict(img, verbose=0)
    return embedding  


def main():

    image_paths = (
        glob.glob(DATA_DIR + "**/*.jpg", recursive=True)
        + glob.glob(DATA_DIR + "**/*.jpeg", recursive=True)
        + glob.glob(DATA_DIR + "**/*.png", recursive=True)
    )
    print(f"Found {len(image_paths)} images.")

    feature_list = []
    for img_path in tqdm(image_paths, desc="Extracting embeddings"):
        feature_list.append(extract_embedding(img_path))

    feature_list = np.squeeze(np.array(feature_list))
    print(f"Feature matrix shape: {feature_list.shape}")

    np.savez(
        OUTPUT_FILE,
        features=feature_list,
        paths=np.array(image_paths),
    )
    print(f"Saved embeddings + paths -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
