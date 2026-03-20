import os                    
import cv2                      
import numpy as np             
import pickle              

from sklearn.neural_network import MLPClassifier      # ANN model
from sklearn.model_selection import train_test_split  
from sklearn.metrics import accuracy_score           

data = []        # Will store image feature vectors
labels = []      # Will store numeric labels for each image
label_map = {}   # Maps numbers → person names

current_label = 0


dataset_path = "../dataset"

print("Loading dataset from:", dataset_path)

# Each folder represents a person

for person in os.listdir(dataset_path):

    person_path = os.path.join(dataset_path, person)

    # Skip files that are not folders
    if not os.path.isdir(person_path):
        continue

    print("Processing images of:", person)

    # Map numeric label to person name
    label_map[current_label] = person

    #  READ ALL IMAGES OF THE PERSON

    for image_name in os.listdir(person_path):

        img_path = os.path.join(person_path, image_name)

        # Read image as grayscale
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        # Resize image so all images have same size
        img = cv2.resize(img, (100, 100))

        #  CONVERT IMAGE INTO FEATURES
        # Flatten converts a 100x100 image into
        # a vector of length 10000

        feature = img.flatten()


        # Store data
        data.append(feature)
        labels.append(current_label)


    current_label += 1

#  CONVERT DATA TO NUMPY

X = np.array(data)
y = np.array(labels)

# SPLIT DATASET
# 80% training
# 20% testing

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# DISPLAY DATASET STATISTICS

print("\nDataset Information : \n")

print("Total images:", len(X))
print("Total persons:", len(label_map))
print("Training images:", len(X_train))
print("Testing images:", len(X_test))


# CREATE ANN MODEL
# MLPClassifier = Multi Layer Perceptron
# hidden_layer_sizes = (128, 64)
# means:
# Layer 1 → 128 neurons
# Layer 2 → 64 neurons

print("\nTraining ANN Model...")

model = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    max_iter=500,
    random_state=42
)

# TRAIN THE MODEL

model.fit(X_train, y_train)

print("Training completed.")

# TEST THE MODEL

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nModel Performance : \n")

print("ANN Accuracy:", accuracy)

# SAVE TRAINED MODEL

model_path = "../models/ann_model.pkl"

with open(model_path, "wb") as f:
    pickle.dump((model, label_map), f)

print("\nModel saved at:", model_path)

# DISPLAY LABEL MAP

print("\nLabel Mapping : \n")

for key, value in label_map.items():
    print(key, "->", value)


print("\nANN Training Completed Successfully.")