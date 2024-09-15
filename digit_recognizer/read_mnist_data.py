import numpy as np
from PIL import Image

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
        img.save(f"training_images/training_image_{i}_{training_labels[i]}.png")


    test_labels = []
    with open(f"{mnist_dir_path}/t10K-labels-idx1-ubyte", 'rb') as f:
        __magic = f.read(4)
        n_items = int.from_bytes(f.read(4), byteorder='big')
        for _ in range(int(n_items)):
            label = int.from_bytes(f.read(1), byteorder='big')
            test_labels.append(label)

    test_images = []
    with open(f"{mnist_dir_path}/t10K-images-idx3-ubyte", 'rb') as f:
        __magic = f.read(4)
        n_items = int.from_bytes(f.read(4), byteorder='big')
        n_rows = int.from_bytes(f.read(4), byteorder='big')
        n_cols = int.from_bytes(f.read(4), byteorder='big')
        for _ in range(int(n_items)):
            image = np.zeros((n_rows, n_cols))
            for i in range(n_rows * n_cols):
                image[i // n_cols][i % n_cols] = int.from_bytes(f.read(1), byteorder='big')

            test_images.append(image)

    # write numpy array as bitmap image to file
    for i, image in enumerate(test_images):
        img = Image.fromarray(image.astype(np.uint8), mode='L')
        img.save(f"test_images/training_image_{i}_{test_labels[i]}.png")


if __name__ == '__main__':
    main()