import os

import numpy as np
import tensorflow as tf
from keras.layers import Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, LeakyReLU, Dropout, BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam  # use from keras.optimizers.legacy import Adam  on MacOS M1
from matplotlib import pyplot as plt

from constants import SPARSE_ARRAY_DATASET_FILE
from data_utils import SparseDataLoader

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def plot_sample_images(batch, title):
    fig, axs = plt.subplots(2, 3)
    for i in range(6):
        row = i // 3  # Determine the row index
        col = i % 3  # Determine the column index
        image_array = batch[i]
        axs[row, col].imshow(image_array, cmap='gray')
        axs[row, col].axis('off')
    plt.tight_layout()  # Adjust the layout to prevent overlapping
    plt.suptitle(title)
    plt.show()


class GAN:
    """
    GAN class encapsulates the generator, discriminator, and GAN models, providing methods for building and compiling each component.
    """

    def __init__(self, latent_dim, data_loader):
        self.latent_dim = latent_dim  # Dimensionality of the latent space
        self.data_loader = data_loader  # Data loader object for loading the dataset
        self.img_shape = (data_loader.image_height, data_loader.image_width, 1)  # Shape of the input images
        self.batch_size = data_loader.batch_size  # Batch size used during training
        self.num_batches = data_loader.num_batches  # Number of batches in the dataset
        self.discriminator = self.build_discriminator()  # Create the discriminator network
        self.generator = self.build_generator()  # Create the generator network
        self.gan_model = self.build_gan()  # Create the GAN model by combining the generator and discriminator

    def build_discriminator(self):
        """
        The build_discriminator method defines the discriminator network architecture.
        """
        model = Sequential([
            Conv2D(16, (self.img_shape[0], 3), strides=(self.img_shape[0], 2), padding='same',
                   input_shape=self.img_shape),
            # Convolutional layer with 16 filters, filter size (height, 3), and stride (height, 2)
            Dropout(0.5),  # Dropout layer to prevent overfitting
            LeakyReLU(alpha=0.2),  # LeakyReLU activation with a negative slope of 0.2
            Dropout(0.5),
            Conv2D(8, (1, 3), strides=(1, 2), padding='same'),
            # Convolutional layer with 8 filters, filter size (1, 3), and stride (1, 2)
            Dropout(0.5),
            LeakyReLU(alpha=0.2),
            Dropout(0.5),
            Flatten(),  # Flatten the output into a 1D vector
            BatchNormalization(),  # Batch normalization layer
            Dropout(0.5),  # Dropout layer
            Dense(1, activation='sigmoid'),  # Dense layer with 1 unit and sigmoid activation for binary classification
        ])
        model.compile(
            loss='binary_crossentropy',
            optimizer=Adam(learning_rate=0.0002, beta_1=0.5),
            metrics=['accuracy']
        )
        return model

    def build_generator(self):
        """
        The build_generator method defines the generator network architecture.
        """
        first_filters_no = 32  # Number of filters in the first layer of the generator
        n_nodes = first_filters_no * int(self.img_shape[0] / 2) * int(
            self.img_shape[1] / 2)  # Number of nodes in the first dense layer
        model = Sequential([
            Dense(n_nodes, input_dim=self.latent_dim),
            # Dense layer with n_nodes units and input dimension as latent_dim
            LeakyReLU(alpha=0.2),
            Reshape((int(self.img_shape[0] / 2), int(self.img_shape[1] / 2), first_filters_no)),
            # Reshape the output into a 4D tensor
            Dense(64),
            LeakyReLU(alpha=0.2),
            Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same'),
            # Transposed convolutional layer with 64 filters, filter size (4, 4), and stride (2, 2)
            Dense(128),
            LeakyReLU(alpha=0.2),
            BatchNormalization(),
            Conv2D(1, (7, 7), padding='same', activation='sigmoid'),
            # Convolutional layer with 1 filter, filter size (7, 7), and sigmoid activation
        ])
        return model

    def build_gan(self):
        """
        The build_gan method builds the GAN model by combining the generator and discriminator networks.
        """
        # Set the discriminator to be non-trainable within the GAN so we can use this model to train the generator
        self.discriminator.trainable = False
        model = Sequential([
            self.generator,  # Generator network
            self.discriminator,  # Discriminator network
        ])
        model.compile(
            loss='binary_crossentropy',
            optimizer=Adam(learning_rate=0.0002, beta_1=0.5),
        )
        return model

    #
    # def get_real_images(self, dataset, n_samples):
    #     ix = np.randint(0, dataset.shape[0], n_samples)
    #     X = dataset[ix]
    #     y = np.ones((n_samples, 1))
    #     return X, y

    def get_latent_data(self, n_samples):
        """
        Generate random latent data for the generator network for each image to be generated.

        Args:
            n_samples (int): Number of latent samples to generate.

        Returns:
            numpy.ndarray: Array of latent data with shape (n_samples, latent_dim).
        """
        # Generate random samples from a standard normal distribution
        x_input = np.random.randn(self.latent_dim * n_samples)
        # Reshape the input into a 2D array of shape (n_samples, latent_dim)
        x_input = x_input.reshape(n_samples, self.latent_dim)
        return x_input

    def get_fake_images(self, n_samples):
        """
        Generate fake images using the generator network.

        Args:
            n_samples (int): Number of fake images to generate.

        Returns:
            tuple: A tuple containing the generated fake images (X) and their labels (y).
                   X: numpy.ndarray of shape (n_samples, image_height, image_width, 1).
                   y: numpy.ndarray of shape (n_samples, 1) with all zeros.
        """
        x_input = self.get_latent_data(n_samples)  # Generate latent data for n_samples
        X = self.generator.predict(x_input)  # Generate fake images using the generator network and the latent data
        y = np.zeros((n_samples, 1))  # Create an array of zeros as the labels for the fake images
        return X, y

    def train(self, n_epochs):
        """
         Train the GAN model for a specified number of epochs.

         Args:
             n_epochs (int): Number of epochs to train the GAN.

         Returns:
             None
         """
        for epoch_no in range(n_epochs):
            print(f"Epoch {epoch_no}")

            # Iterate over each batch in the data loader
            for batch_no, X_real in enumerate(self.data_loader):
                n_samples = X_real.shape[0]

                # Train Discriminator
                y_real = np.ones((n_samples, 1))  # Label real samples as 1 (real)
                X_fake, y_fake = self.get_fake_images(n_samples=n_samples)  # Generate fake images and labels
                # Concatenate real and fake samples along with their labels
                X, y = np.vstack((X_real, X_fake)), np.vstack((y_real, y_fake))
                # Train the discriminator on the combined real and fake samples
                d_loss, _ = self.discriminator.train_on_batch(X, y)

                # Train Generator
                X_gan = self.get_latent_data(n_samples=n_samples)  # Generate latent data for the generator
                # Label the generated samples as 1 (real) to fool the discriminator
                y_gan = np.ones((n_samples, 1))
                # Train the GAN model (generator) using the latent data and target labels
                g_loss = self.gan_model.train_on_batch(X_gan, y_gan)

                # Plot fake samples every 5th batch
                if batch_no % 5 == 0:
                    plot_sample_images(batch=X_fake, title=f'Fake Samples epoch {epoch_no}')

                print(
                    f'Epoch {epoch_no + 1}/{n_epochs}, Batch {batch_no}/{self.num_batches}, dloss={round(d_loss, 3)}, gloss={round(g_loss, 3)}')


def main(batch_size=64):
    """
    batch_size: Number of samples in each training batch
    """
    LATENT_DIM = 100  # Size of the latent space (1D vector of random numbers that is used to initiate the generator)
    IMAGE_HEIGHT, IMAGE_WIDTH = (88, 112)  # Dimensions of the generated images
    EPOCHS = 3  # Number of training epochs
    tf.keras.backend.clear_session()  # Clears the previous TensorFlow session (Clear RAM memory)
    # Create a SparseDataLoader object for loading the MIDI piano roll dataset
    print("Creating SparseDataLoader")
    data_loader = SparseDataLoader(
        file_name=SPARSE_ARRAY_DATASET_FILE,  # File name of the dataset
        batch_size=batch_size,  # Set the batch size for loading data
        image_width=IMAGE_WIDTH,  # Set the width of the generated images
        image_height=IMAGE_HEIGHT  # Set the height of the generated images
    )
    # Create a GAN (Generative Adversarial Network) object
    print("Creating GAN")
    gan = GAN(
        latent_dim=LATENT_DIM,  # Set the dimensionality of the latent space
        data_loader=data_loader,  # Provide the data loader for training
    )
    # Train the GAN
    print("Training GAN")
    gan.train(
        n_epochs=EPOCHS  # Set the number of training epochs
    )
    print("done")


if __name__ == "__main__":
    main()
