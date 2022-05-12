# Custom  L1 Distance layer module

# Import dependencies
import tensorflow as tf
from tensorflow.keras.layers import Layer

# Custom L1 Distance Layer from Jupyter
# Why is this Important: Its needed to load the custom model
# From Module 4


## Build Distance Layer

# creating distance layer class
# subtracting twin datasets from each other to determine level similarity
# output= anchor - (positive or negative )

class L1Dist(Layer):
    # using python inheritance method using Init

    def __init__(self, **kwargs):
        super().__init__()

    # magic happens here- similarity calculation
    # imput embedding = anchor
    # validation_embedding = positive or negative datasets

    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)