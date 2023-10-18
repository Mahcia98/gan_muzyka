from pathlib import Path

MAESTRO_DATA_URL = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip"

DATA_DIR_PATH = Path('data')
DATA_DIR_PATH.mkdir(exist_ok=True)

MAESTR_ZIP_FILE_PATH = DATA_DIR_PATH / "maestro-v3.0.0-midi.zip"
MAESTRO_DATASET_PATH = DATA_DIR_PATH / 'maestro-v3.0.0'
SPARSE_ARRAY_DATASET_FILE = DATA_DIR_PATH / 'training_dataset.npz'

MIDI_PATH = Path('midi')
MIDI_PATH.mkdir(exist_ok=True)