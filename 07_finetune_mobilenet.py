import tensorflow as tf
import numpy as np
import os
from sklearn.utils.class_weight import compute_class_weight

# -------------------------------
# SETTINGS
# -------------------------------

dataset_path = "dataset"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
EPOCHS = 5

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

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(labels),
    y=labels
)

class_weights = dict(enumerate(weights))

print("\nClass weights:")

for i, weight in class_weights.items():
    print(class_names[i], ":", round(weight, 2))

# -------------------------------
# LOAD EXISTING MODEL
# -------------------------------

model = tf.keras.models.load_model(
    "models/mobilenetv2_waste.keras"
)

print("\nExisting MobileNetV2 model loaded.")

# -------------------------------
# FIND MOBILENETV2 BASE
# -------------------------------

base_model = None

for layer in model.layers:
    if isinstance(layer, tf.keras.Model):
        base_model = layer
        break

if base_model is None:
    raise ValueError("MobileNetV2 base model not found.")

print("Base model found:", base_model.name)

# -------------------------------
# FINE-TUNING
# -------------------------------

base_model.trainable = True

# Freeze most layers
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Keep BatchNormalization layers frozen
for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

print("\nFine-tuning last 30 MobileNetV2 layers.")

# -------------------------------
# COMPILE
# -------------------------------

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.00001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# -------------------------------
# CALLBACKS
# -------------------------------

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "models/mobilenetv2_finetuned.keras",
    monitor="val_accuracy",
    save_best_only=True
)

# -------------------------------
# TRAIN
# -------------------------------

print("\nStarting fine-tuning...")

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=[
        early_stopping,
        checkpoint
    ]
)

print("\nFine-tuning completed!")

print(
    "Model saved as:"
)

print(
    "models/mobilenetv2_finetuned.keras"
)