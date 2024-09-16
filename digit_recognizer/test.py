import argparse
import numpy as np

from mnist_decoder import MNISTDecoder
from neural_network.neuralnet import NeuralNetwork

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--idx", type=int, default=-1)

    args = parser.parse_args()

    n_inputs = 28 * 28
    n_outputs = 10
    nn = NeuralNetwork([n_inputs, 1000, 100, 50, n_outputs])
    nn.load_weights("digit_recognizer/weights/1_epoch")

    mnist_data = "mnist_dataset"
    test_decoder = MNISTDecoder(
        f"{mnist_data}/t10k-images-idx3-ubyte",
        f"{mnist_data}/t10k-labels-idx1-ubyte"
    )
    test_decoder.init()

    if args.idx != -1:
        image, label = test_decoder.get_image_and_label_at(args.idx)
        normalized_image = image.astype(np.float32) / 255
        output = nn.forward(list(normalized_image.flat))
        guess = output.index(max(output))
        print("guess was:", guess)
        print("label was:", label)
        return

    correct = 0
    for _ in range(test_decoder.n_items):
        image, label = test_decoder.get_next_image_and_label()
        normalized_image = image.astype(np.float32) / 255
        output = nn.forward(list(normalized_image.flat))
        guess = output.index(max(output))
        if guess == label:
            correct += 1
    print(f"Accuracy: {correct/test_decoder.n_items}")

if __name__ == '__main__':
    main()