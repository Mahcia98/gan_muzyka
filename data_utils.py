import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
from scipy.ndimage import zoom

from constants import SPARSE_ARRAY_DATASET_FILE


class SparseDataLoader:
    """
    DataLoader for sparse matrix with long piano roll matrix in sparse format.
    It yields batches whenever a new batch is needed.
    """

    def __init__(self, file_name, batch_size, image_width, image_height):
        """
        Initialize the SparseDataLoader.

        Args:
            file_name (str): File name/path of the sparse matrix.
            batch_size (int): Size of each batch.
            image_width (int): Width of each image in the batch.
            image_height (int): Height of each image in the batch.
        """
        self.sparse_matrix = sc.sparse.load_npz(file_name)
        self.batch_size = batch_size
        self.image_width = image_width
        self.image_height = image_height
        self.num_batches = self.sparse_matrix.shape[0] // batch_size

    def __iter__(self):
        """
        Iterates over the DataLoader to yield batches of images.

        Yields:
            numpy.ndarray: Batch of images.
        """
        for batch_index in range(self.num_batches):
            # Calculate start and end indices for the current batch
            start_index = batch_index * self.image_width * self.batch_size
            end_index = min((batch_index + 1) * self.image_width * self.batch_size, self.sparse_matrix.shape[0])

            # Check if the batch is at the end of the data
            if start_index >= self.sparse_matrix.shape[0]:
                break

            # Extract only this part of the sparse matrix that will be used in the current batch
            rows_to_extract = np.arange(start_index, end_index)
            batch_sparse_matrix = self.sparse_matrix[rows_to_extract, :]

            # Transform sparse matrix to dense form
            batch_dense_matrix = batch_sparse_matrix.toarray()

            # Resize the batch to match the target shape
            resized_batch = np.zeros((self.batch_size, self.image_height, self.image_width, 1))

            for i in range(self.batch_size):
                resized_batch[i, :, :, 0] = zoom(batch_dense_matrix[i, :, :, 0], (
                self.image_height / batch_dense_matrix.shape[1], self.image_width / batch_dense_matrix.shape[2]))

            # Reshape batch as needed
            batch_images = batch_dense_matrix.reshape(self.batch_size, self.image_width, self.image_height, 1)
            batch_images = np.transpose(batch_images, (0, 2, 1, 3))
            yield batch_images

        #for batch_index in range(self.num_batches):
        #    start_index = batch_index * self.image_width * self.batch_size
        #   end_index = (batch_index + 1) * self.image_width * self.batch_size

        #    if end_index > self.sparse_matrix.shape[0]:
        #        print("Invalid batch indices:", start_index, end_index)


if __name__ == "__main__":

    BATCH_SIZE = 1000000

    IMAGE_HEIGHT, IMAGE_WIDTH = (88, 112)

    data_loader = SparseDataLoader(
        file_name='data/training_dataset_bool_resolution_by_4.npz',
        # SPARSE_ARRAY_DATASET_FILE,
        batch_size=BATCH_SIZE,
        image_width=IMAGE_WIDTH,
        image_height=IMAGE_HEIGHT
    )

    # Plot single image from batch loaded using SparseDataLoader class
    for batch in data_loader:
        print(batch.shape)
        # plt.plot(batch.mean(axis=(0, 2, 3)));plt.show()  # plot average keys stroke in dataset (use big BATCH_SIZE>200k)
        image_array = batch[0]
        plt.imshow(image_array, cmap='gray')
        plt.axis('off')
        plt.show()
        break
