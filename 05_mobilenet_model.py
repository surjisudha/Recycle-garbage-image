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

print("Classes:")
print(class_names)

# -------------------------------
# CLASS WEIGHTS
# -------------------------------

class_counts = []

for class_name in class_names:
    class_path = os.path.join(dataset_path, class_name)
    class_counts.append(len(os.listdir(class_path)))

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
# DATA AUGMENTATION
# -------------------------------

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2)
])

# -------------------------------
# MOBILE NET V2
# -------------------------------

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze pretrained layers
base_model.trainable = False

# -------------------------------
# BUILD MODEL
# -------------------------------

inputs = tf.keras.Input(shape=(224, 224, 3))

x = data_augmentation(inputs)

# MobileNetV2 preprocessing
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

x = base_model(x, training=False)

x = tf.keras.layers.GlobalAveragePooling2D()(x)

x = tf.keras.layers.Dense(128, activation="relu")(x)

x = tf.keras.layers.Dropout(0.5)(x)

outputs = tf.keras.layers.Dense(6, activation="softmax")(x)

model = tf.keras.Model(inputs, outputs)

# -------------------------------
# COMPILE
# -------------------------------

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# -------------------------------
# MODEL SUMMARY
# -------------------------------

model.summary()

# -------------------------------
# TRAIN
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

model.save("models/mobilenetv2_waste.keras")

print("\nMobileNetV2 training completed!")
print("Model saved as:")
print("models/mobilenetv2_waste.keras")