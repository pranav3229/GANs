import os
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib as plt
import cv2

# Assuming you have datasets of original and corrupted images
# Load the original and corrupted images using your preferred method

# Preprocess the images
# Normalize pixel values to be in the range [-1, 1]
def preprocess_images(images):
    return (np.array(images) - 127.5) / 127.5



from tensorflow.keras import models, layers

import tensorflow as tf
from tensorflow.keras import layers, models

# Function to build the Generator model
def build_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(8 * 8 * 256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((8, 8, 256)))
    assert model.output_shape == (None, 8, 8, 256)  # Note: None is the batch size

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 16, 16, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 32, 32, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 64, 64, 3)

    return model

# Function to build the Discriminator model
def build_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[64, 64, 3]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

# Build the GAN
def build_gan(generator, discriminator):
    discriminator.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True))
    discriminator.trainable = False

    gan = models.Sequential()
    gan.add(generator)
    gan.add(discriminator)
    gan.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True))
    return gan






# Train the GAN
def train_gan(original_images, corrupted_images, epochs, batch_size):
    # Preprocess the images
    original_images = preprocess_images(original_images)
    corrupted_images = preprocess_images(corrupted_images)

    generator = build_generator_model()
    discriminator = build_discriminator_model()
    gan = build_gan(generator, discriminator)

    for epoch in range(epochs):
        for i in range(0, len(original_images), batch_size):
            real_images = original_images[i:i + batch_size]
            fake_images = generator.predict(np.random.randn(batch_size, 100))  # Random noise for generating fake images

            # Train the discriminator
            d_loss_real = discriminator.train_on_batch(real_images, np.ones((len(real_images), 1)))
            d_loss_fake = discriminator.train_on_batch(fake_images, np.zeros((batch_size, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Train the generator
            noise = np.random.randn(batch_size, 100)  # Generate new noise
            gan_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))

            # Print progress
            print(f"Epoch: {epoch+1}, Discriminator Loss: {d_loss}, GAN Loss: {gan_loss}")

# Paths to the directories containing the original and corrupt images
original_images_path = '/home/postman/dl_project_tanna/train+val/train/Earthquake'
corrupt_images_path = '/home/postman/dl_project_tanna/train+val/train/Earthquake_paired'

# Function to load images from a directory
def load_images_from_directory(directory):
    images = []
    for filename in os.listdir(directory):
        img = cv2.imread(os.path.join(directory, filename))
        images.append(img)
    return images

# Load original and corrupt images
original_images = load_images_from_directory(original_images_path)
corrupt_images = load_images_from_directory(corrupt_images_path)

generator=build_generator_model()
# Now, you can use the provided GAN code by passing the loaded original and corrupt images to the `train_gan` function
# Ensure the paths are correctly set and the images are loaded in a format suitable for the GAN architecture.
train_gan(original_images, corrupt_images, epochs=10, batch_size=32)


# Function to remove black lines from a specific corrupt image using the trained generator
def remove_black_lines(generator, corrupt_image_path):
    # Load the corrupt image
    corrupt_image = cv2.imread(corrupt_image_path)
    corrupt_image = cv2.cvtColor(corrupt_image, cv2.COLOR_BGR2RGB)
    corrupt_image = cv2.resize(corrupt_image, (32, 32))  # Assuming the GAN was trained on 32x32 images

    # Preprocess the corrupt image
    corrupt_image = np.array([corrupt_image])
    corrupt_image = (corrupt_image - 127.5) / 127.5

    # Use the generator to remove the black lines
    generated_image = generator.predict(corrupt_image)
    generated_image = (generated_image * 127.5 + 127.5).astype(np.uint8)
    generated_image = np.squeeze(generated_image)

    return generated_image

# Example usage
corrupt_image_path = '/home/postman/dl_project_tanna/train+val/train/Earthquake_paired/E1 (2).png'
generated_image = remove_black_lines(generator, corrupt_image_path)

# Display the original and generated images for comparison
original_image = cv2.imread('/home/postman/dl_project_tanna/train+val/train/Earthquake/E1 (2).png')
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(original_image)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Generated Image (Black Lines Removed)')
plt.imshow(generated_image)
plt.axis('off')

plt.tight_layout()
plt.show()
