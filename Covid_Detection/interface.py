import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Load the saved model
model = load_model('best_model.keras')

# Define the image paths and their corresponding labels
image_paths = [
    'chest_xray/COVID/images/COVID-9.png',
    'chest_xray/COVID/images/COVID-27.png',
    'chest_xray/Lung_Opacity/images/Lung_Opacity-1.png',
    'chest_xray/Lung_Opacity/images/Lung_Opacity-120.png',
    'test/NORMAL/NORMAL2-IM-0051-0001.jpeg',
    'test/NORMAL/IM-0005-0001.jpeg',
    'test/PNEUMONIA/person1_virus_8.jpeg',
    'test/PNEUMONIA/person92_bacteria_451.jpeg'
]

labels = ['COVID', 'COVID', 'Lung Opacity', 'Lung Opacity', 'Normal', 'Normal', 'Pneumonia', 'Pneumonia']
class_labels = {0: 'COVID', 1: 'Lung Opacity', 2: 'Normal', 3: 'Viral Pneumonia'}

# Calculate rows and columns for subplot
num_images = len(image_paths)
cols = 4  # You can adjust this number based on your screen size
rows = num_images // cols + (num_images % cols > 0)

fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(cols * 5, rows * 5))
fig.suptitle('Prediction Results', fontsize=16)

axes = axes.flatten()  # Flatten the axes array for easier indexing

# Loop through each image, process, predict, and display
for idx, (path, true_label) in enumerate(zip(image_paths, labels)):
    try:
        img = image.load_img(path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize pixel values

        predictions = model.predict(img_array)
        predicted_class = class_labels[np.argmax(predictions)]

        # Display the image and its prediction
        ax = axes[idx]
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f'Predicted: {predicted_class}\nTrue: {true_label}\nConfidence: {np.max(predictions):.2f}')
    except Exception as e:
        print(f"Error processing image {path}: {str(e)}")
        ax.set_title(f'Error loading image')
        ax.axis('off')

# Turn off unused axes
for i in range(idx + 1, len(axes)):
    axes[i].axis('off')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the layout to make room for the main title
plt.show()
