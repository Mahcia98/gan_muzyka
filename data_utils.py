import matplotlib.pyplot as plt
import numpy as np
import scipy as sc

from constants import SPARSE_ARRAY_DATASET_FILE


class SparseDataLoader:
    def __init__(self, file_name, batch_size, image_width, image_height):
        self.sparse_matrix = sc.sparse.load_npz(file_name)
        self.batch_size = batch_size
        self.image_width = image_width
        self.image_height = image_height
        self.num_batches = self.sparse_matrix.shape[0] // batch_size

    def __iter__(self):
        for batch_index in range(self.num_batches):
            start_index = batch_index * self.image_width * self.batch_size
            end_index = (batch_index + 1) * self.image_width * self.batch_size

            rows_to_extract = np.arange(start_index, end_index)
            batch_sparse_matrix = self.sparse_matrix[rows_to_extract, :]

            batch_dense_matrix = batch_sparse_matrix.toarray()
            batch_images = batch_dense_matrix.reshape(self.batch_size, self.image_width, self.image_height, 1)
            batch_images = np.transpose(batch_images, (0, 2, 1, 3))
            yield batch_images


if __name__ == "__main__":

    BATCH_SIZE = 64

    IMAGE_HEIGHT, IMAGE_WIDTH = (88, 112)

    data_loader = SparseDataLoader(
        file_name=SPARSE_ARRAY_DATASET_FILE,
        batch_size=BATCH_SIZE,
        image_width=IMAGE_WIDTH,
        image_height=IMAGE_HEIGHT
    )

    for batch in data_loader:
        print(batch.shape)

        image_array = batch[0]
        plt.imshow(image_array, cmap='gray')
        plt.axis('off')
        plt.show()
        break
