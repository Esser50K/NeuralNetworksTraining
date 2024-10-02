import argparse
import numpy as np

from mnist_decoder import MNISTDecoder
from neural_network.neuralnet import NeuralNetwork
from PIL import Image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--idx", type=int, default=-1)
    parser.add_argument("--path", type=str, default="")

    args = parser.parse_args()

    n_inputs = 28 * 28
    n_outputs = 10
    nn = NeuralNetwork([n_inputs, 100, 100, n_outputs], learning_rate=0.01)
    nn.load_weights("digit_recognizer/weights/2_epoch")

    # load the image from a path and test it
    if args.path != "":
        image = Image.open(args.path)
        image = image.convert("L")
        image = np.array(image)
        normalized_image = image.astype(np.float32) / 255
        output = nn.forward(list(normalized_image.flat))
        guess = output.index(max(output))
        print("guess was:", guess)
        return

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

    total_avg_confidence = 0
    avg_confidence_per_output = [0] * 10
    test_input_count = [0] * 10
    correct_outputs_by_label = [0] * 10
    correct = 0
    for _ in range(test_decoder.n_items):
        image, label = test_decoder.get_next_image_and_label()
        normalized_image = image.astype(np.float32) / 255
        output = nn.forward(list(normalized_image.flat))
        guess = output.index(max(output))
        if guess == label:
            correct += 1
            correct_outputs_by_label[label] += 1

        total_avg_confidence += output[label]
        avg_confidence_per_output[label] += output[label]

        test_input_count[label] += 1

    total_avg_confidence /= test_decoder.n_items
    for i in range(10):
        avg_confidence_per_output[i] /= test_input_count[i]

    print(f"Accuracy: {correct/test_decoder.n_items}")
    print("Avg Confidence:", total_avg_confidence)
    print("Test Distribution:", test_input_count)
    print("Correct Outputs By Label:", correct_outputs_by_label)
    print("Avg Confidence Per Output:", avg_confidence_per_output)

if __name__ == '__main__':
    main()