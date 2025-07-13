import cv2
import dlib
import os
import numpy as np
from sklearn.cluster import DBSCAN

# Setup paths (paths will be provided as per your project requirements)
folder_path = r"C:\Users\Hashir Irfan\OneDrive\Desktop\NEURAL NETWORK\CNN\DATA\ALL IMAGES"  # Path to the folder containing images
storing_folder = r"C:\Users\Hashir Irfan\OneDrive\Desktop\NEURAL NETWORK\CNN\DATA\ALL_FACES"  # Path where cropped images will be saved
face_folders = r"C:\Users\Hashir Irfan\OneDrive\Desktop\NEURAL NETWORK\CNN\DATA\Segmented_Faces"  # Path where similar faces will be grouped

# Initialize the dlib face detector and the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r"C:\Users\Hashir Irfan\AppData\Local\Programs\Python\Python312\shape_predictor_68_face_landmarks.dat")  # Download this model from dlib
embedder = dlib.face_recognition_model_v1(r"C:\Users\Hashir Irfan\AppData\Local\Programs\Python\Python312\dlib_face_recognition_resnet_model_v1.dat")  # Download this model from dlib

# Function to extract embeddings from a face (image)
def extract_face_embedding(image, face_rect):
    shape = predictor(image, face_rect)
    embedding = np.array(embedder.compute_face_descriptor(image, shape))
    return embedding

# Function to process and save cropped face images
def process_image(image_path):
    # Read the image using OpenCV
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the image
    faces = detector(gray_image)
    face_embeddings = []
    face_crops = []
    
    for i, face in enumerate(faces):
        # Extract the face embedding (representation) using dlib
        embedding = extract_face_embedding(image, face)
        face_embeddings.append(embedding)
        
        # Crop the face from the image
        face_crop = image[face.top():face.bottom(), face.left():face.right()]
        
        # Resize the cropped face to 150x150 resolution
        resized_face = cv2.resize(face_crop, (200, 200))
        
        # Create a new folder for each face (if not exists already)
        face_image_filename = os.path.join(storing_folder, f"face_{os.path.basename(image_path)}_{i}.jpg")
        cv2.imwrite(face_image_filename, resized_face)
        
        face_crops.append(face_image_filename)
    
    return face_embeddings, face_crops

# Step 1: Process all images in the folder
face_embeddings_all = []
face_crops_all = []

for img_file in os.listdir(folder_path):
    img_path = os.path.join(folder_path, img_file)
    if os.path.isfile(img_path):
        embeddings, crops = process_image(img_path)
        face_embeddings_all.extend(embeddings)
        face_crops_all.extend(crops)

print(f"Processed {len(face_crops_all)} cropped faces.")



# Step 2: Cluster faces using DBSCAN and organize them into folders based on similarity

# Convert embeddings to a numpy array for clustering
face_embeddings_array = np.array(face_embeddings_all)

# Use DBSCAN to cluster similar faces (DBSCAN does not require you to specify the number of clusters)
dbscan = DBSCAN(eps=0.6, min_samples=5, metric='euclidean')  # eps is the distance threshold
labels = dbscan.fit_predict(face_embeddings_array)

# Step 3: Organize faces into folders based on clustering results
for label in set(labels):
    if label == -1:
        continue  # Ignore noise points in DBSCAN
    # Create a folder for each label (cluster)
    cluster_folder = os.path.join(face_folders, f"cluster_{label}")
    os.makedirs(cluster_folder, exist_ok=True)
    
    # Move all images corresponding to this cluster into the folder
    for i, label_assigned in enumerate(labels):
        if label_assigned == label:
            cropped_face_path = face_crops_all[i]
            # Move the cropped face image to the corresponding cluster folder
            new_path = os.path.join(cluster_folder, os.path.basename(cropped_face_path))
            os.rename(cropped_face_path, new_path)

print(f"Organized faces into {len(set(labels))} clusters.")