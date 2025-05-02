import os
import numpy as np
from PIL import Image
from scipy.ndimage import convolve, uniform_filter
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from scipy.special import expit as sigmoid

# Constants
WINDOW_SIZE = 15

# ========== Filters ==========
def outer_product(f1, f2):
    return np.outer(f1, f2)

def generate_laws_kernels():
    L5 = [1, 4, 6, 4, 1]
    E5 = [-1, -2, 0, 2, 1]
    S5 = [-1, 0, 2, 0, -1]
    W5 = [-1, 2, 0, -2, 1]
    R5 = [1, -4, 6, -4, 1]
    vectors = [L5, E5, S5, W5, R5]
    return [outer_product(f1, f2) for f1 in vectors for f2 in vectors]

# ========== Processing Steps ==========
def read_grayscale_image(path):
    img = Image.open(path).convert("L")
    return np.array(img, dtype=np.float32)

def convolve_image(image, kernel):
    return np.abs(convolve(image, kernel, mode="reflect"))

def mean_blur(image, size):
    return uniform_filter(image, size=size, mode="reflect")

def extract_features(image, kernels):
    return [mean_blur(convolve_image(image, k), WINDOW_SIZE) for k in kernels]

def flatten_features(feature_maps):
    stacked = np.stack(feature_maps, axis=-1)
    h, w, d = stacked.shape
    vectors = stacked.reshape(-1, d)
    return vectors, h, w

def cluster_and_color(vectors, h, w, k):
    # Normalize with sigmoid
    vectors = sigmoid(vectors)
    
    # KMeans clustering
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(vectors)

    # Map each label to a color
    label_image = labels.reshape(h, w)
    return label_to_color_image(label_image, k)

def label_to_color_image(label_img, k):
    colors = np.random.randint(0, 255, size=(k, 3), dtype=np.uint8)
    h, w = label_img.shape
    color_image = np.zeros((h, w, 3), dtype=np.uint8)

    for i in range(k):
        color_image[label_img == i] = colors[i]
    return color_image

# ========== Wrapper ==========
def process_image(path, k):
    image = read_grayscale_image(path)
    kernels = generate_laws_kernels()
    features = extract_features(image, kernels)
    vectors, h, w = flatten_features(features)
    return cluster_and_color(vectors, h, w, k)
