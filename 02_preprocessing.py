import tensorflow as tf
import os
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Dataset location
dataset_path = "dataset"

# Image settings
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

# Load training data
train_data = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# Load validation data
val_data = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# Class names
class_names = train_data.class_names

print("Class names:")
print(class_names)

# Count images in each class
class_counts = []

for class_name in class_names:
    class_path = os.path.join(dataset_path, class_name)
    count = len(os.listdir(class_path))
    class_counts.append(count)

print("\nClass counts:")
for i in range(len(class_names)):
    print(class_names[i], ":", class_counts[i])

# Create labels for each image
labels = []

for class_index, count in enumerate(class_counts):
    labels.extend([class_index] * count)

labels = np.array(labels)

# Calculate class weights
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(labels),
    y=labels
)

class_weights = dict(enumerate(class_weights))

print("\nClass weights:")

for i, weight in class_weights.items():
    print(class_names[i], ":", round(weight, 2))

print("\nPreprocessing completed successfully!")