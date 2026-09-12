import tensorflow as tf
import os
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

# -------------------------------
# SETTINGS
# -------------------------------

dataset_path = "dataset"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
EPOCHS = 10

# -------------------------------
# LOAD DATA
# -------------------------------

train_data = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_data = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_data.class_names

print("Classes:", class_names)

# -------------------------------
# NORMALIZATION
# -------------------------------

normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)

# -------------------------------
# DATA AUGMENTATION
# -------------------------------

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2)
])

# -------------------------------
# CLASS WEIGHTS
# -------------------------------

class_counts = []

for class_name in class_names:
    class_path = os.path.join(dataset_path, class_name)
    count = len(os.listdir(class_path))
    class_counts.append(count)

labels = []

for class_index, count in enumerate(class_counts):
    labels.extend([class_index] * count)

labels = np.array(labels)

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(labels),
    y=labels
)

class_weights = dict(enumerate(class_weights_array))

print("\nClass weights:")
for i, weight in class_weights.items():
    print(class_names[i], ":", round(weight, 2))

# -------------------------------
# BUILD CNN MODEL
# -------------------------------

model = tf.keras.Sequential([

    tf.keras.Input(shape=(224, 224, 3)),

    data_augmentation,

    normalization_layer,

    # First convolution block
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    # Second convolution block
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    # Third convolution block
    tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    # Flatten
    tf.keras.layers.Flatten(),

    # Dense layer
    tf.keras.layers.Dense(128, activation="relu"),

    # Dropout
    tf.keras.layers.Dropout(0.5),

    # Output layer
    tf.keras.layers.Dense(6, activation="softmax")
])

# -------------------------------
# COMPILE MODEL
# -------------------------------

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Display model architecture
model.summary()

# -------------------------------
# TRAIN MODEL
# -------------------------------

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    class_weight=class_weights
)

# -------------------------------
# SAVE MODEL
# -------------------------------

os.makedirs("models", exist_ok=True)

model.save("models/baseline_cnn.keras")

print("\nCNN training completed!")
print("Model saved as:")
print("models/baseline_cnn.keras")