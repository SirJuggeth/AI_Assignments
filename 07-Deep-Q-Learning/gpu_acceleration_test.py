import tensorflow as tf

# Check the version of TensorFlow being used
print("\nTensorFlow version:", tf.__version__)

# Check if TensorFlow can access the GPU
print("\nIs GPU available:", tf.test.is_gpu_available())
print("Num GPUs Available:", len(tf.config.list_physical_devices("GPU")))
print("GPU list: ", tf.config.list_physical_devices("GPU"))

# Get detailed GPU info
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    details = tf.config.experimental.get_device_details(gpu)
    print(details)

print("\nCUDA version:", tf.sysconfig.get_build_info()["cuda_version"])
print("cuDNN version:", tf.sysconfig.get_build_info()["cudnn_version"])
