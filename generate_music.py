import click
from mido import Message, MidiFile, MidiTrack

from constants import SPARSE_ARRAY_DATASET_FILE, MIDI_PATH
from data_utils import SparseDataLoader
from train_model import MODELS_PATH, GAN


def roll_to_track(roll):
    def stop_note(note, time):
        return Message('note_off', note=note, velocity=0, time=time)

    def start_note(note, time):
        return Message('note_on', note=note, velocity=127, time=time)

    delta = 0
    # State of the notes in the roll.
    notes = [False] * len(roll[0])
    # MIDI note for first column.
    midi_base = 0
    for row in roll:
        for i, col in enumerate(row):
            note = midi_base + i
            if col == 1:
                if notes[i]:
                    # First stop the ringing note
                    yield stop_note(note, delta)
                    delta = 0
                yield start_note(note, delta)
                delta = 0
                notes[i] = True
            elif col == 0:
                if notes[i]:
                    # Stop the ringing note
                    yield stop_note(note, delta)
                    delta = 0
                notes[i] = False
        # ms per row
        delta += 500


def generate_midi_from_array(arr, name='sample'):
    midi = MidiFile(type=1)
    midi.tracks.append(MidiTrack(roll_to_track(arr.T)))
    midi_file_path = MIDI_PATH / f'{name}.mid'
    midi.save(midi_file_path)


def generate(load_from_path, name):
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
    X_fake, _ = gan.get_fake_images(n_samples=1)
    arr = X_fake[0].reshape(X_fake.shape[1], X_fake.shape[2]).astype(int)
    print(arr.shape)
    generate_midi_from_array(arr=arr, name=name)


@click.command()
@click.option('--load_from_path', type=str, help='Path to load from (string)')
@click.option('--name', type=str, help='name of output file (string)')
def cli(name, load_from_path):
    """
    Usage:
        python generate_music.py --load_from_path 'gan_save_2023_10_18' --name 'sample'
    """
    generate(name=name, load_from_path=MODELS_PATH / load_from_path)


if __name__ == "__main__":
    cli()
