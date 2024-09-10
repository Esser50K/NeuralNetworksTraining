from neuralnet import NeuralNetwork

def main():
    nn = NeuralNetwork([3, 2, 2])
    input_arr = [1, 0, 0]
    # output = nn.forward(input_arr)
    # print("output", output, type(output))

    print("matrix before training", nn)
    nn.train(input_arr, [1, 0])
    print("matrix after training", nn)

if __name__ == '__main__':
    main()