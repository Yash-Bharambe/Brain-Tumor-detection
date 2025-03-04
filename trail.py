import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split


IMG_SIZE = (128, 128)
BATCH_SIZE = 32


data_dir = 'path_to_your_dataset_directory'  # Update this path
yes_dir = os.path.join(data_dir, 'yes')
no_dir = os.path.join(data_dir, 'no')

images = []
labels = []


for category in ['yes', 'no']:
    class_num = 1 if category == 'yes' else 0
    category_dir = os.path.join(data_dir, category)
    for img in os.listdir(category_dir):
        try:
            img_path = os.path.join(category_dir, img)
            img = load_img(img_path, target_size=IMG_SIZE)
            img_array = img_to_array(img)
            images.append(img_array)
            labels.append(class_num)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")


images = np.array(images)
labels = np.array(labels)


images = images / 255.0


x_train, x_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)


train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator()

train_generator = train_datagen.flow(x_train, y_train, batch_size=BATCH_SIZE)
val_generator = val_datagen.flow(x_val, y_val, batch_size=BATCH_SIZE)


model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=len(x_train) // BATCH_SIZE,
    epochs=25,
    validation_data=val_generator,
    validation_steps=len(x_val) // BATCH_SIZE
)


loss, accuracy = model.evaluate(val_generator, steps=len(x_val) // BATCH_SIZE)
print(f'Validation Accuracy: {accuracy * 100:.2f}%')


plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# Save the model
model.save('brain_tumor_detection_model.h5')
