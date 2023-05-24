from constants import MAESTRO_DATASET_PATH, SPARSE_ARRAY_DATASET_FILE
from midi_utils import MIDITransformer

if __name__ == "__main__":
    print('Generating Training Dataset')
    midi_transformer = MIDITransformer(midi_file_dir_path=MAESTRO_DATASET_PATH)
    midi_transformer.generate_training_dataset(save_path=SPARSE_ARRAY_DATASET_FILE)
