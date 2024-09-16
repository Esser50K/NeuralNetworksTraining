from typing import Optional

import numpy as np
from PIL import Image


def label_offset(idx: int) -> int:
    return 8 + idx

def image_offset(idx: int) -> int:
    return 8 + (idx * 28 * 28)

class MNISTDecoder:
    def __init__(self, images_path: str, labels_path: str):
        self.images_path = images_path
        self.labels_path = labels_path

        self.images_file = None
        self.labels_file = None
        self.n_items = 0
        self.current_item = 0

    def init(self):
        self.images_file = open(self.images_path, 'rb')
        self.labels_file = open(self.labels_path, 'rb')

        self.images_file.seek(4)
        self.n_items = int.from_bytes(self.images_file.read(4), byteorder='big')

    def get_next_image_and_label(self) -> tuple[Optional[np.ndarray], Optional[int]]:
        if self.current_item >= self.n_items:
            return None, None

        self.images_file.seek(image_offset(self.current_item))
        self.labels_file.seek(label_offset(self.current_item))
        label = int.from_bytes(self.labels_file.read(1), byteorder='big')
        image = np.zeros((28, 28))
        for i in range(28 * 28):
            image[i // 28][i % 28] = int.from_bytes(self.images_file.read(1), byteorder='big')

        self.current_item += 1
        return image, label

    def get_image_and_label_at(self, idx: int) -> tuple[Optional[np.ndarray], Optional[int]]:
        self.images_file.seek(image_offset(idx))
        self.labels_file.seek(label_offset(idx))
        label = int.from_bytes(self.labels_file.read(1), byteorder='big')
        image = np.zeros((28, 28))
        for i in range(28 * 28):
            image[i // 28][i % 28] = int.from_bytes(self.images_file.read(1), byteorder='big')

        self.current_item += 1
        return image, label


def main():
    mnist_dir_path = "../mnist_dataset"

    training_labels = []
    with open(f"{mnist_dir_path}/train-labels-idx1-ubyte", 'rb') as f:
        __magic = f.read(4)
        n_items = int.from_bytes(f.read(4), byteorder='big')
        for _ in range(int(n_items)):
            label = int.from_bytes(f.read(1), byteorder='big')
            training_labels.append(label)

    training_images = []
    with open(f"{mnist_dir_path}/train-images-idx3-ubyte", 'rb') as f:
        __magic = f.read(4)
        n_items = int.from_bytes(f.read(4), byteorder='big')
        n_rows = int.from_bytes(f.read(4), byteorder='big')
        n_cols = int.from_bytes(f.read(4), byteorder='big')
        for _ in range(int(n_items)):
            image = np.zeros((n_rows, n_cols))
            for i in range(n_rows * n_cols):
                image[i // n_cols][i % n_cols] = int.from_bytes(f.read(1), byteorder='big')

            training_images.append(image)

    # write numpy array as bitmap image to file
    for i, image in enumerate(training_images):
        img = Image.fromarray(image.astype(np.uint8), mode='L')
        img.save(f"images/training_image_{i}_{training_labels[i]}.png")


if __name__ == '__main__':
    main()