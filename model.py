import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
import numpy as np
from sklearn.metrics import classification_report, accuracy_score

# Define paths to the training and validation datasets
train_data_dir = '/home/postman/dl_project_tanna/impainted_dataset/train'
validation_data_dir = '/home/postman/dl_project_tanna/impainted_dataset/validation'

# Image dimensions
img_width, img_height = 256, 256  

# Define parameters
batch_size = 10
epochs = 70
num_classes = 5

# Data augmentation and normalization with more augmentation techniques
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=[0.5, 1.5],
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))  # Adjust the dropout rate as needed
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))  # Adjust the dropout rate as needed
model.add(Dense(num_classes, activation='softmax'))

for layer in base_model.layers:
    layer.trainable = True

model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Remove early stopping and reduce LR callbacks
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)

validation_preds = model.predict(validation_generator)

class_labels = list(validation_generator.class_indices.keys())
true_labels = validation_generator.classes
predicted_labels = np.argmax(validation_preds, axis=1)

accuracy = accuracy_score(true_labels, predicted_labels)
report = classification_report(true_labels, predicted_labels, target_names=class_labels)

print("Accuracy: {:.2f}%".format(accuracy * 100))
print("Class-wise Metrics:")
print(report)

# Calculate class-wise accuracy
class_accuracy = []
for cls in class_labels:
    cls_idx = class_labels.index(cls)
    cls_true_labels = true_labels == cls_idx
    cls_pred_labels = predicted_labels == cls_idx
    cls_acc = accuracy_score(cls_true_labels, cls_pred_labels)
    class_accuracy.append(cls_acc)

class_accuracy_dict = dict(zip(class_labels, class_accuracy))
print("Class-wise Accuracy:")
print(class_accuracy_dict)
