import os
import platform
from datetime import datetime
from pathlib import Path

import click
import numpy as np
import tensorflow as tf
from keras.layers import Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, LeakyReLU, Dropout, BatchNormalization
from keras.models import Sequential
from keras.src.saving.saving_lib import load_model

from constants import SPARSE_ARRAY_DATASET_FILE
from data_utils import SparseDataLoader

if platform.system() == 'Windows':
    from keras.optimizers import Adam
elif platform.system() == 'Darwin':
    from keras.optimizers.legacy import Adam
else:
    from keras.optimizers import Adam

from matplotlib import pyplot as plt
from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

IMAGES_PATH = Path('images')
MODELS_PATH = Path('models')
MODELS_PATH.mkdir(exist_ok=True)
GENERATED_IMAGES_PATH = IMAGES_PATH / 'generated_images'
GENERATED_IMAGES_PATH.mkdir(exist_ok=True, parents=True)

current_date_hour = datetime.now().strftime("%Y_%m_%d_%H_%M")


def plot_sample_images(batch, title, batch_epoch_str, single_image=False, save_images=False):
    batch = batch.astype(int) * 255
    if single_image:
        # TODO: save image and create alternative version for 1 image
        pass
    else:
        fig, axs = plt.subplots(2, 3)
        fig.subplots_adjust(wspace=0.03, hspace=0.1)
        for i in range(6):
            row = i // 3  # Determine the row index
            col = i % 3  # Determine the column index
            image_array = batch[i]
            axs[row, col].imshow(image_array, cmap='gray', aspect='auto')
            axs[row, col].axis('off')
    # plt.tight_layout()  # Adjust the layout to prevent overlapping
    plt.suptitle(f"{title} | {batch_epoch_str}")
    if save_images:
        # TODO: path with date, batch_epoch_str and add folder generated_images
        fig.savefig(GENERATED_IMAGES_PATH / f'image_{current_date_hour}_{batch_epoch_str}.png')
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

    def save_model(self, path: Path):
        """
        Save the GAN model to a file.

        Args:
            path (Path): The file path where the model will be saved.
        """
        path.mkdir(exist_ok=True)
        discriminator_path = path / "discriminator.keras"
        generator_path = path / "generator.keras"

        self.discriminator.save(discriminator_path)
        self.generator.save(generator_path)

    def load_model(self, path: Path):
        """
        Load a GAN model from a file.

        Args:
            path (Path): The file path from which the model will be loaded.

        Returns:
            GAN: An instance of the GAN class with the loaded model and data loader.
        """
        discriminator_path = path / "discriminator.keras"
        generator_path = path / "generator.keras"

        self.discriminator = load_model(
            discriminator_path,
            custom_objects={"BatchNormalization": BatchNormalization}
        )
        self.generator = load_model(
            generator_path,
            custom_objects={"BatchNormalization": BatchNormalization}
        )
        # latent_dim = loaded_model.layers[0].input_shape[1]
        self.gan_model = self.build_gan()

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
        X = self.generator.predict(x_input,
                                   verbose=0)  # Generate fake images using the generator network and the latent data
        X = (X > 0.5)  # I'm using thresholding to convert float dtype to bool. Input images have bool dtype as well
        y = np.zeros((n_samples, 1))  # Create an array of zeros as the labels for the fake images
        return X, y

    def train(self, n_epochs, n_batch):
        """
         Train the GAN model for a specified number of epochs.

         Args:
             n_epochs (int): Number of epochs to train the GAN.
             n_batch

         Returns:
             None
         """
        for epoch_no in range(n_epochs):
            # Iterate over each batch in the data loader
            pbar = tqdm(enumerate(self.data_loader), total=self.num_batches)
            for batch_no, X_real in pbar:
                if batch_no > n_batch:
                    break
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

                if batch_no == 0:
                    plot_sample_images(
                        batch=X_real,
                        title=f'Real Samples',
                        batch_epoch_str=f"{batch_no}_{self.num_batches} epoch {epoch_no}",
                        save_images=True
                    )
                # Plot fake samples every n-th batch
                if batch_no % 20 == 0:
                    plot_sample_images(
                        batch=X_fake,
                        title=f'Fake Samples',
                        batch_epoch_str=f"{batch_no}_{self.num_batches} epoch {epoch_no}",
                        save_images=True
                    )

                pbar.set_description(
                    f'Epoch {epoch_no + 1}_{n_epochs}, d_loss={round(d_loss, 3)}, g_loss={round(g_loss, 3)}'
                )


def main(batch_size=128, image_height=88, image_width=112, epochs=1, n_batch=20, load_from_path=None):
    """
    batch_size: Number of samples in each training batch
    image_height, image_width: Dimensions of the generated images
    epochs: Number of training epochs
    """
    LATENT_DIM = 100  # Size of the latent space (1D vector of random numbers that is used to initiate the generator)
    tf.keras.backend.clear_session()  # Clears the previous TensorFlow session (Clear RAM memory)
    # Create a SparseDataLoader object for loading the MIDI piano roll dataset
    print("Creating SparseDataLoader")
    data_loader = SparseDataLoader(
        file_name=SPARSE_ARRAY_DATASET_FILE,  # File name of the dataset
        batch_size=batch_size,  # Set the batch size for loading data
        image_width=image_width,  # Set the width of the generated images
        image_height=image_height  # Set the height of the generated images
    )
    # Create a GAN (Generative Adversarial Network) object
    print("Creating GAN")
    gan = GAN(
        latent_dim=LATENT_DIM,  # Set the dimensionality of the latent space
        data_loader=data_loader,  # Provide the data loader for training
    )
    if load_from_path is not None:
        print("Loading Model")
        gan.load_model(load_from_path)
    # Train the GAN
    print("Training GAN")
    gan.train(n_epochs=epochs, n_batch=n_batch)
    print("Save model")
    gan.save_model(path=MODELS_PATH / f"gan_save_{current_date_hour}")


@click.command()
@click.option('--epochs', type=int, default=1, help='Number of epochs (integer)')
@click.option('--n_batch', type=int, default=1, help='Number of batch (integer)')
@click.option('--load_from_path', type=str, default=None, help='Path to load from (string)')
def cli(n_batch, epochs, load_from_path):
    """
    Usage:
        python train_model.py --epochs 1 --n_batch 200
        python train_model.py --epochs 1 --n_batch 200 --load_from_path 'gan_save_2023_10_18'
    """
    load_from_path = MODELS_PATH/load_from_path if load_from_path else None
    main(n_batch=n_batch, epochs=epochs, load_from_path=load_from_path)


if __name__ == "__main__":
    cli()

