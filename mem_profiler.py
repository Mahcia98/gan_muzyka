# pip install memory_profiler

from memory_profiler import profile

from constants import SPARSE_ARRAY_DATASET_FILE
from data_utils import SparseDataLoader


@profile
def profile_data_loader():
    BATCH_SIZE = 512
    IMAGE_HEIGHT, IMAGE_WIDTH = (88, 112)
    data_loader = SparseDataLoader(
        file_name=SPARSE_ARRAY_DATASET_FILE,
        batch_size=BATCH_SIZE,
        image_width=IMAGE_WIDTH,
        image_height=IMAGE_HEIGHT
    )
    for batch in data_loader:
        print(batch.shape)
        break


if __name__ == "__main__":
    profile_data_loader()
