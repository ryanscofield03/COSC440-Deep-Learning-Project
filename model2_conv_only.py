import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from geometry import TIGREDataset
import skimage.io


class Model2Conv(tf.keras.Model):
    """
    A model class for Attenuation coefficient prediction from https://arxiv.org/abs/2209.14540
    This implementation uses an argument encoder to encode points in 3-dimensional space and
    then passes the encoding to several dense layers to produce the predicted attenuation
    at that point in 3-dimensional space.
    """

    def __init__(self, encoder, bound=0.2, num_layers=4, hidden_dim=124, skips=[1], out_dim=1,
                 last_activation="sigmoid", filters=32, height=32, width=32):
        super(Model2Conv, self).__init__()

        self.encoder = encoder
        self.bound = bound
        self.in_dim = self.encoder.get_output_dim()  # Get the input dimension from the encoder
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.skips = skips
        self.out_dim = out_dim
        self.height = height
        self.width = width

        self.conv_feature_dim = 32
        self.skip_size = self.conv_feature_dim + self.in_dim

        self.conv_block = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters, 9, padding="SAME"),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            tf.keras.layers.Conv2D(filters * 2, 5, padding="SAME"),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            tf.keras.layers.Conv2D(filters * 4, 3, padding="SAME"),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(self.conv_feature_dim)
        ])

        # Define the layers
        self.mlp_layers = []
        # First layer
        self.mlp_layers.append(tf.keras.layers.Dense(hidden_dim))

        # Intermediate layers
        for i in range(1, num_layers - 1):
            if i in skips:
                self.mlp_layers.append(tf.keras.layers.Dense(hidden_dim + self.skip_size))
            else:
                self.mlp_layers.append(tf.keras.layers.Dense(hidden_dim))

        # Output layer
        self.mlp_layers.append(tf.keras.layers.Dense(out_dim))

        # Activation functions
        self.activations = []
        for i in range(num_layers - 1):
            self.activations.append(tf.keras.layers.LeakyReLU(alpha=0.2))  # Equivalent to nn.LeakyReLU() in PyTorch

        # Handle last activation
        if last_activation == "sigmoid":
            self.activations.append(tf.keras.layers.Activation("sigmoid"))
        elif last_activation == "relu":
            self.activations.append(tf.keras.layers.LeakyReLU(alpha=0.2))
        else:
            raise NotImplementedError("Unknown last activation")

    def call(self, x, projection_image):
        # First, encode the input using the encoder
        x = self.encoder(x)

        proj = tf.reshape(projection_image, [1, self.height, self.width, 1])
        conv_features = self.conv_block(proj)
        conv_features_tiled = tf.tile(conv_features, [tf.shape(x)[0], 1])

        conv_features_tiled = tf.keras.layers.LayerNormalization()(conv_features_tiled)

        x = tf.concat([x, conv_features_tiled], axis=1)
        input_pts = x

        # Apply the layers
        for i in range(self.num_layers):
            layer = self.mlp_layers[i]
            activation = self.activations[i] if i < len(self.activations) else None

            # If this layer is a skip layer, concatenate the input points
            if i in self.skips:
                x = tf.concat([input_pts, x], axis=-1)

            # Apply the linear transformation
            x = layer(x)

            # Apply the activation function
            if activation:
                x = activation(x)

        return x
