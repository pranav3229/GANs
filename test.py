import tensorflow as tf

# Check for GPU availability
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

# Check if TensorFlow is using GPU
print("Is TensorFlow using GPU?", tf.test.is_built_with_cuda())

# Check the cuDNN version
print("CuDNN Version:", tf.config.list_physical_devices('GPU'))

# Get detailed GPU information for each available GPU
for gpu in tf.config.list_physical_devices('GPU'):
    print(tf.config.experimental.get_device_details(gpu))
