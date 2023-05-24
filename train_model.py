import os

import numpy as np
import tensorflow as tf
from keras.layers import Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, LeakyReLU, Dropout, BatchNormalization
from keras.models import Sequential
from keras.optimizers.legacy import Adam
from matplotlib import pyplot as plt

from constants import SPARSE_ARRAY_DATASET_FILE
from data_utils import SparseDataLoader

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


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
    def __init__(self, latent_dim, data_loader):
        self.latent_dim = latent_dim
        self.data_loader = data_loader
        self.img_shape = (data_loader.image_height, data_loader.image_width, 1)
        self.batch_size = data_loader.batch_size
        self.num_batches = data_loader.num_batches
        self.discriminator = self.build_discriminator()
        self.generator = self.build_generator()
        self.gan_model = self.build_gan()

    def build_discriminator(self):
        model = Sequential([
            Conv2D(16, (self.img_shape[0], 3), strides=(self.img_shape[0], 2), padding='same', input_shape=self.img_shape),
            Dropout(0.5),
            LeakyReLU(alpha=0.2),
            Dropout(0.5),
            Conv2D(8, (1, 3), strides=(1, 2), padding='same'),
            Dropout(0.5),
            LeakyReLU(alpha=0.2),
            Dropout(0.5),
            Flatten(),
            BatchNormalization(),
            Dropout(0.5),
            Dense(1, activation='sigmoid'),
        ])
        model.compile(
            loss='binary_crossentropy',
            optimizer=Adam(learning_rate=0.0002, beta_1=0.5),
            metrics=['accuracy']
        )
        return model

    def build_generator(self):
        first_filters_no = 32
        n_nodes = first_filters_no * int(self.img_shape[0] / 2) * int(self.img_shape[1] / 2)
        model = Sequential([
            Dense(n_nodes, input_dim=self.latent_dim),
            LeakyReLU(alpha=0.2),
            Reshape((int(self.img_shape[0] / 2), int(self.img_shape[1] / 2), first_filters_no)),
            Dense(64),
            LeakyReLU(alpha=0.2),
            Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same'),
            Dense(128),
            # LeakyReLU(alpha=0.2),
            # Dense(1024),
            LeakyReLU(alpha=0.2),
            BatchNormalization(),
            Conv2D(1, (7, 7), padding='same', activation='sigmoid'),
        ])
        return model

    def build_gan(self):
        self.discriminator.trainable = False
        model = Sequential([
            self.generator,
            self.discriminator,
        ])
        model.compile(
            loss='binary_crossentropy',
            optimizer=Adam(learning_rate=0.0002, beta_1=0.5)
        )
        return model

    #
    # def get_real_images(self, dataset, n_samples):
    #     ix = np.randint(0, dataset.shape[0], n_samples)
    #     X = dataset[ix]
    #     y = np.ones((n_samples, 1))
    #     return X, y

    def get_latent_data(self, n_samples):
        x_input = np.random.randn(self.latent_dim * n_samples)
        x_input = x_input.reshape(n_samples, self.latent_dim)
        return x_input

    def get_fake_images(self, n_samples):
        x_input = self.get_latent_data(n_samples)
        X = self.generator.predict(x_input)
        y = np.zeros((n_samples, 1))
        return X, y

    def train(self, n_epochs):
        for epoch_no in range(n_epochs):
            print(f"Epoch {epoch_no}")
            for batch_no, X_real in enumerate(self.data_loader):
                n_samples = X_real.shape[0]
                print('Train Discriminator')
                y_real = np.ones((n_samples, 1))
                # if epoch_no == 0 and batch_no == 0: # Plot Real Only First Tile
                #     plot_sample_images(batch=X_real, title='Real Samples')
                X_fake, y_fake = self.get_fake_images(n_samples=n_samples)
                X, y = np.vstack((X_real, X_fake)), np.vstack((y_real, y_fake))
                d_loss, _ = self.discriminator.train_on_batch(X, y)
                print('Train Generator')
                X_gan = self.get_latent_data(n_samples=n_samples)
                y_gan = np.ones((n_samples, 1))
                g_loss = self.gan_model.train_on_batch(X_gan, y_gan)
                if batch_no % 5 == 0:  # Plot fake at the beginning of the epoch
                    plot_sample_images(batch=X_fake, title=f'Fake Samples epoch {epoch_no}')
                print(
                    f'Epoch {epoch_no + 1}/{n_epochs}, Batch {batch_no}/{self.num_batches}, dloss={round(d_loss, 3)}, gloss={round(g_loss, 3)}'
                )


def main():
    BATCH_SIZE = 64
    LATENT_DIM = 100
    IMAGE_HEIGHT, IMAGE_WIDTH = (88, 112)
    EPOCHS = 3

    tf.keras.backend.clear_session()
    print("Creating SparseDataLoader")
    data_loader = SparseDataLoader(
        file_name=SPARSE_ARRAY_DATASET_FILE,
        batch_size=BATCH_SIZE,
        image_width=IMAGE_WIDTH,
        image_height=IMAGE_HEIGHT
    )

    print("Creating GAN")
    gan = GAN(
        latent_dim=LATENT_DIM,
        data_loader=data_loader,
    )
    print("Training GAN")
    gan.train(
        n_epochs=EPOCHS
    )
    print("done")


if __name__ == "__main__":
    main()
