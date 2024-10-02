import numpy as np

from mnist_decoder import MNISTDecoder
from neural_network.neuralnet import NeuralNetwork

def main():
    n_inputs = 28 * 28
    n_outputs = 10
    nn = NeuralNetwork([n_inputs, 100, 100, n_outputs])
    # nn.load_weights("digit_recognizer/weights/1_epoch")

    mnist_data = "mnist_dataset"
    training_decoder = MNISTDecoder(
        f"{mnist_data}/train-images-idx3-ubyte",
        f"{mnist_data}/train-labels-idx1-ubyte"
    )
    training_decoder.init()

    print(f"Training on {training_decoder.n_items} images")
    epochs = 2
    for epoch in range(epochs):
        for i in range(training_decoder.n_items):
            image, label = training_decoder.get_image_and_label_at(i)
            output = [0] * 10
            output[label] = 1

            normalized_image = image.astype(np.float32) / 255
            try:
                nn.train(list(normalized_image.flat), output)
            except Exception as e:
                print(f"Error on image {i} with label {label}:", e)
            if i % 1000 == 0:
                print(f"Trained {i} images")
    nn.store_weights(f"digit_recognizer/weights/{epochs}_epoch")


if __name__ == '__main__':
    main()