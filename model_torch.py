import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import timm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Define paths to the training and validation datasets
train_data_dir = '/home/postman/dl_project_tanna/impainted_dataset/train'
validation_data_dir = '/home/postman/dl_project_tanna/impainted_dataset/validation'

# Image dimensions
img_width, img_height = 256, 256  

# Define parameters
batch_size = 10
epochs = 20  # Increase the number of epochs for more training
num_classes = 5  # Number of classes (earthquake, wildfire, flood, landslides, hurricane)

# Data augmentation and normalization with more augmentation techniques

# timm
efficientnet_b7 = timm.create_model('efficientnet_b7', pretrained=True)
efficientnet_b7 = tf.keras.applications.efficientnet.preprocess_input

# COlor jittering
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

# Use timm model as base_model
base_model = Sequential([
    efficientnet_b7,
    GlobalAveragePooling2D(),
])

# Add custom classification layers on top of EfficientNetB7
model = Sequential([
    base_model,
    Dense(1024, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# Unfreeze the last layers for fine-tuning
for layer in base_model.layers[-20:]:
    layer.trainable = True

# Compile the model with a lower learning rate and using ReduceLROnPlateau
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    callbacks=[early_stopping, reduce_lr]
)

# Predict on validation data using the best weights obtained from early stopping
validation_preds = model.predict(validation_generator)

# Rest of your evaluation code remains the same
class_labels = list(validation_generator.class_indices.keys())
true_labels = validation_generator.classes
predicted_labels = np.argmax(validation_preds, axis=1)

accuracy = accuracy_score(true_labels, predicted_labels)
report = classification_report(true_labels, predicted_labels, target_names=class_labels)

print("Accuracy: {:.2f}%".format(accuracy * 100))
print("Class-wise Metrics:")
print(report)
