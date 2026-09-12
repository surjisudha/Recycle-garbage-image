import os
import matplotlib.pyplot as plt

dataset_path = "dataset"

classes = []
counts = []

for class_name in os.listdir(dataset_path):
    class_path = os.path.join(dataset_path, class_name)

    if os.path.isdir(class_path):
        classes.append(class_name)
        counts.append(len(os.listdir(class_path)))

print("Waste Categories:")
print(classes)

print("\nNumber of images in each category:")

for i in range(len(classes)):
    print(classes[i], ":", counts[i])

# Bar chart
plt.figure(figsize=(10, 6))
plt.bar(classes, counts)

plt.title("Number of Images in Each Waste Category")
plt.xlabel("Waste Category")
plt.ylabel("Number of Images")

plt.xticks(rotation=45)
plt.tight_layout()

plt.show()
import random
from PIL import Image

# Display one sample image from each category
plt.figure(figsize=(15, 8))

for i, class_name in enumerate(classes):
    class_path = os.path.join(dataset_path, class_name)

    images = os.listdir(class_path)

    # Select one random image
    image_name = random.choice(images)
    image_path = os.path.join(class_path, image_name)

    image = Image.open(image_path)

    plt.subplot(2, 3, i + 1)
    plt.imshow(image)
    plt.title(class_name)
    plt.axis("off")

plt.tight_layout()
plt.show()