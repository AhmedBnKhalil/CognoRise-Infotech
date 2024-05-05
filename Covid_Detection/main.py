# import Deep learning Libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Ignore Warnings
import warnings

warnings.filterwarnings("ignore")


def create_datagen():
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=20,
        zoom_range=0.2,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        validation_split=0.2  # Use 20% of the data for validation
    )

    train_generator = train_datagen.flow_from_directory(
        directory='chest_xray',  # Update with your path
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    validation_generator = train_datagen.flow_from_directory(
        directory='chest_xray',  # Update with your path
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )

    return train_generator, validation_generator


def build_model():
    base_model = tf.keras.applications.VGG19(include_top=False, weights="imagenet", input_shape=(224, 224, 3),
                                             pooling='max')
    base_model.trainable = False  # Freeze the base model initially

    model = Sequential([
        base_model,
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(4, activation='softmax')  # Assuming 4 classes
    ])

    model.compile(optimizer=Adamax(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model


def train_model(train_gen, val_gen):
    model = build_model()

    checkpoint = ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True, verbose=1)
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.2, patience=2, min_lr=0.0001, verbose=1)

    history = model.fit(
        train_gen,
        steps_per_epoch=train_gen.samples // train_gen.batch_size,
        validation_data=val_gen,
        validation_steps=val_gen.samples // val_gen.batch_size,
        epochs=20,
        callbacks=[checkpoint, early_stopping, reduce_lr]
    )
    return model, history


def evaluate_model(model, validation_generator):
    # Evaluate the model on the validation set
    results = model.evaluate(validation_generator, steps=len(validation_generator))
    print(f"Validation Loss: {results[0]:.4f}, Validation Accuracy: {results[1]:.4f}")


def predict_new_image(model, image_path):
    # Load and preprocess the image
    img = load_img(image_path, target_size=(224, 224))  # Load and resize the image
    img_array = img_to_array(img)  # Convert the image to numpy array
    img_array = img_array / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add a batch dimension

    # Make the prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)  # Get the class with the highest probability

    # Optional: Mapping the class indices back to class labels, if you have that information
    class_labels = {0: 'COVID', 1: 'Lung Opacity', 2: 'Normal', 3: 'Viral Pneumonia'}
    predicted_label = class_labels[predicted_class[0]]

    # Display the results
    plt.imshow(img)
    plt.title(f'Predicted: {predicted_label}, Confidence: {np.max(predictions):.2f}')
    plt.axis('off')
    plt.show()

    return predicted_label, np.max(predictions)


def main():
    train_gen, val_gen = create_datagen()
    model, history = train_model(train_gen, val_gen)
    print("Model training complete.")

    # Load the best model saved by ModelCheckpoint
    evaluate_model(model, val_gen)


if __name__ == "__main__":
    main()
