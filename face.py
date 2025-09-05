import cv2
import os
from retinaface import RetinaFace
from deepface import DeepFace

# Directories
input_dir = './input'
faces_dir = './output'
known_faces_dir = './known_faces'

# Create necessary directories if they don't exist
os.makedirs(faces_dir, exist_ok=True)
os.makedirs(known_faces_dir, exist_ok=True)

# Clear the faces directory to make it volatile
for file in os.listdir(faces_dir):
    file_path = os.path.join(faces_dir, file)
    if os.path.isfile(file_path):
        os.remove(file_path)

# Load known faces and their paths
known_faces = {}
for person_folder in os.listdir(known_faces_dir):
    person_path = os.path.join(known_faces_dir, person_folder)
    if os.path.isdir(person_path):
        for filename in os.listdir(person_path):
            known_face_path = os.path.join(person_path, filename)
            known_faces[known_face_path] = person_folder

# Step 1: Detect faces in input images and save cropped faces to the faces directory
for filename in os.listdir(input_dir):
    img_path = os.path.join(input_dir, filename)
    img = cv2.imread(img_path)
    
    # Check if file is a valid image
    if img is None:
        print(f"Skipping {filename} (not a valid image file)")
        continue

    # Detect faces with RetinaFace
    faces = RetinaFace.detect_faces(img)

    for i, (key, face) in enumerate(faces.items()):
        # Extract bounding box coordinates
        facial_area = face["facial_area"]
        x, y, w, h = facial_area

        # Crop the face
        cropped_face = img[y:h, x:w]
        
        # Save the cropped face temporarily in the faces directory for recognition
        face_filename = os.path.join(faces_dir, f"{os.path.splitext(filename)[0]}_face_{i+1}.jpg")
        cv2.imwrite(face_filename, cropped_face)

        # Step 2: Compare with known faces using DeepFace
        found_match = False
        for known_face_path, person_name in known_faces.items():
            try:
                result = DeepFace.verify(img1_path=face_filename, img2_path=known_face_path, model_name='VGG-Face', enforce_detection=False)
                if result["verified"]:
                    print(f"Face already known: {person_name}")
                    person_folder = os.path.join(known_faces_dir, person_name)
                    found_match = True
                    break
            except Exception as e:
                print(f"Error comparing faces: {e}")

        # If no match is found, prompt for a new name
        if not found_match:
            name = input(f"Enter name for {face_filename}: ")
            person_folder = os.path.join(known_faces_dir, name)
            os.makedirs(person_folder, exist_ok=True)
            known_faces[face_filename] = name

        # Save the new face in the corresponding person's folder
        face_save_path = os.path.join(person_folder, os.path.basename(face_filename))
        cv2.imwrite(face_save_path, cropped_face)

print("Face detection, cropping, and organized labeling completed.")
