# 1. install requiremnents
''''
pip install -r requiremnts.txt
'''


# 2. Test if GPU is working...
'''
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("Available GPUs:", tf.config.list_physical_devices('GPU'))

# Create a simple matrix multiplication to test GPU
with tf.device('/GPU:0'):
    a = tf.random.normal([1000, 1000])
    b = tf.random.normal([1000, 1000])
    c = tf.matmul(a, b)

print("Matrix multiplication complete!")
'''

# if It has "Matrix multiplication complete!" at the end then it works and your device is all set!


# 3. Check if PyTorch's Metal backend (MPS) is available for GPU acceleration on macOS.
'''
import torch
print(torch.backends.mps.is_available())  # Should print True
'''
# if it has "True" at the end then it works and you are ready to start!
