from scipy import sparse

from constants import MAESTRO_DATASET_PATH, SPARSE_ARRAY_DATASET_FILE
from midi_utils import load_midi_files, generate_training_dataset, generate_training_dataset_multiprocessing

if __name__ == "__main__":
    print('Generating Training Dataset')
    midi_file_path_list = load_midi_files(path=MAESTRO_DATASET_PATH)
    try:
        result_array_concatenated = generate_training_dataset_multiprocessing(file_path_list=midi_file_path_list)
    except Exception as e:
        print(f"Error running with Multiprocessing: {e}\nRunning on single core instead.")
        result_array_concatenated = generate_training_dataset(file_path_list=midi_file_path_list)
    print("Saving the results")
    sparse.save_npz(SPARSE_ARRAY_DATASET_FILE, result_array_concatenated)
