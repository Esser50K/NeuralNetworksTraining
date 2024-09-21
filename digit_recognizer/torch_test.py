import argparse
import numpy as np
import torch

from mnist_decoder import MNISTDecoder
from torch_neural_network.cnn import CNN
from PIL import Image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--idx", type=int, default=-1)
    parser.add_argument("--path", type=str, default="")

    args = parser.parse_args()

    cnn = CNN()
    cnn.load_state_dict(torch.load("digit_recognizer/torch_weights/2_epoch", weights_only=False))
    cnn.eval()

    # # load the image from a path and test it
    # if args.path != "":
    #     image = Image.open(args.path)
    #     image = image.convert("L")
    #     image = np.array(image)
    #     normalized_image = image.astype(np.float32) / 255
    #     output = cnn(normalized_image)
    #     guess = output.index(max(output))
    #     print("guess was:", guess)
    #     return
    #
    mnist_data = "mnist_dataset"
    test_decoder = MNISTDecoder(
        f"{mnist_data}/t10k-images-idx3-ubyte",
        f"{mnist_data}/t10k-labels-idx1-ubyte"
    )
    test_decoder.init()
    #
    # if args.idx != -1:
    #     image, label = test_decoder.get_image_and_label_at(args.idx)
    #     normalized_image = image.astype(np.float32) / 255
    #     output = cnn.forward(normalized_image)
    #     guess = output.index(max(output))
    #     print("guess was:", guess)
    #     print("label was:", label)
    #     return

    total_avg_confidence = 0
    avg_confidence_per_output = [0] * 10
    test_input_count = [0] * 10
    correct_outputs_by_label = [0] * 10
    correct = 0
    for _ in range(test_decoder.n_items):
        image, label = test_decoder.get_next_image_and_label()
        normalized_image = image.astype(np.float32) / 255

        # Convert to tensor and add batch and channel dimensions
        # go from (28, 28) to (1, 1, 28, 28)
        image_tensor = torch.tensor(normalized_image).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            output = cnn(image_tensor)

        probabilities = torch.softmax(output, dim=1)
        guess = torch.argmax(probabilities, dim=1).item()
        if guess == label:
            correct += 1
            correct_outputs_by_label[label] += 1

        # Accumulate confidence metrics
        total_avg_confidence += probabilities[0][label].item()
        avg_confidence_per_output[label] += probabilities[0][label].item()
        test_input_count[label] += 1

    total_avg_confidence /= test_decoder.n_items
    for i in range(10):
        if test_input_count[i] > 0:
            avg_confidence_per_output[i] /= test_input_count[i]

    print(f"Accuracy: {correct/test_decoder.n_items}")
    print("Test Distribution:", test_input_count)
    print("Avg Confidence:", total_avg_confidence)
    print("Avg Confidence Per Output:", avg_confidence_per_output)
    print("Correct Outputs By Label:", correct_outputs_by_label)

if __name__ == '__main__':
    main()