import click

from constants import SPARSE_ARRAY_DATASET_FILE
from data_utils import SparseDataLoader
from train_model import MODELS_PATH, GAN


def generate(load_from_path, n_samples=1):
    print("Creating SparseDataLoader")
    data_loader = SparseDataLoader(
        file_name=SPARSE_ARRAY_DATASET_FILE,  # File name of the dataset
        batch_size=128, image_height=88, image_width=112
    )
    # Create a GAN (Generative Adversarial Network) object
    print("Creating GAN")
    LATENT_DIM = 100  # Size of the latent space (1D vector of random numbers that is used to initiate the generator)
    gan = GAN(
        latent_dim=LATENT_DIM,  # Set the dimensionality of the latent space
        data_loader=data_loader,  # Provide the data loader for training
    )
    if load_from_path is not None:
        print("Loading Model")
        gan.load_model(load_from_path)
    # Generate Music Samples
    X_fake, _ = gan.get_fake_images(n_samples=n_samples)
    print(X_fake.shape)


@click.command()
@click.option('--n_samples', type=int, default=1, help='Number of sample (integer)')
@click.option('--load_from_path', type=str, help='Path to load from (string)')
def cli(n_samples, load_from_path):
    """
    Usage:
        python generate_music.py --n_samples 1 --load_from_path 'gan_save_2023_10_18'
    """
    generate(n_samples=n_samples, load_from_path=MODELS_PATH/load_from_path)


if __name__ == "__main__":
    cli()