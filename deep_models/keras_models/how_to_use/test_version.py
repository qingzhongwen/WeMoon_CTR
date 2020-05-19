# import os
# from tensorflow.python.client import device_lib
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "99"
#
# if __name__ == "__main__":
#     print(device_lib.list_local_devices())

import tensorflow as tf
tf.config.list_physical_devices('GPU')
tf.test.is_gpu_available()