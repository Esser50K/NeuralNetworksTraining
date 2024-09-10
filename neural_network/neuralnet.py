import math

import numpy as np
from neural_network.matrix import Matrix

class NeuralNetwork:
    def __init__(self, layers: list[int], learning_rate = 0.01) -> None:
        # weights can be represented as matrix with the columns equal to the number of the input layer
        # and the rows equal to the number of the output layer
        self.layer_weights = [
            Matrix(next_layer, layer, randomize=True) for layer, next_layer in zip(layers, layers[1:])
        ]

        self.layer_bias_weights = [
            Matrix(layer, 1, randomize=True) for layer in layers[1:]
        ]

        self.learning_rate = learning_rate

    def __str__(self) -> str:
        return "\n".join([str(layer) for layer in self.layer_weights])

    def forward(self, inputs: list[float]) -> list[float]:
        output_matrix, _ = self.__internal_forward(inputs)

        # flatten the output and return it without the bias
        return list(output_matrix.data.flat)

    def __internal_forward(self, inputs: list[float]) -> tuple[Matrix, list[Matrix]]:
        # create matrix for inputs for easy multiplication
        inputs_matrix = Matrix(len(inputs), 1)
        inputs_matrix.data = np.array(inputs).reshape(-1, 1)
        # print("input shape", inputs_matrix.data, inputs_matrix.data.shape)

        layer_outputs = [inputs_matrix]
        for layer, weights in enumerate(self.layer_weights):
            # print("layer", layer)
            # # add bias by adding a row with a 1
            # inputs_matrix.data = np.vstack((inputs_matrix.data, np.ones((inputs_matrix.data.shape[1], 1))))
            # print("weight shape", weights.data.shape)
            # print("input shape", inputs_matrix.data.shape)
            inputs_matrix = weights.multiply_matrix(inputs_matrix)
            # add bias
            inputs_matrix = inputs_matrix.add_matrix(self.layer_bias_weights[layer])
            # print("output shape", inputs_matrix.data.shape)

            # apply activation function, a sigmoid
            inputs_matrix = inputs_matrix.apply_fn(lambda x: 1 / (1 + pow(math.e, -x)))
            layer_outputs.append(inputs_matrix)

        # since we manipulate the inputs_matrix internally it has become the output
        return inputs_matrix, layer_outputs

    def train(self, inputs: list[float], targets: list[float]) -> None:
        # feedforward
        # output = self.forward(inputs)
        # output_matrix = Matrix(1, len(output))
        # output_matrix.data[0] = output
        output_matrix, layer_outputs = self.__internal_forward(inputs)
        # print("layer outputs", len(layer_outputs))

        targets_matrix = Matrix(len(targets), 1)
        targets_matrix.data = np.array(targets).reshape(-1, 1)

        # print("backpropagation start")
        # back-propagation algorithm
        # calculate the error: error = target - output
        # print("output", output_matrix.data, output_matrix.data.shape)
        # print("target", targets_matrix.data, targets_matrix.data.shape)

        error_matrix = targets_matrix.subtract_matrix(output_matrix)  # output layer error matrix
        # print("error", error_matrix.data)

        for i in range(len(self.layer_weights)-1, -1, -1):
            # print("\n")
            # print("backprop layer", i)
            # calculate the gradient: how much the weights should change
            # ΔW = α * ∂E/∂W
            # ΔW = lr * E * (O * (1 - O)) * transpose(I)
            # delta of weights equals the learning rate times the error times the gradient times the input

            # (O * (1 - O))
            gradient = layer_outputs[i+1].apply_fn(lambda x: x * (1 - x))
            # E * (O * (1 - O)) -> delta of error for each weight
            gradient = gradient.element_wise_multiply(error_matrix)
            # lr * E * (O * (1 - O)) -> we multiply by learning rate to nudge the weights in the right direction but not fully correct
            gradient = gradient.multiply_scalar(self.learning_rate)

            # calculate the weight deltas
            transposed_layer = layer_outputs[i].transpose()
            # print("gradient", gradient.data, gradient.data.shape)
            # print("transposed layer", transposed_layer.data, transposed_layer.data.shape)
            weight_deltas = gradient.multiply_matrix(transposed_layer)
            # weight_deltas.data = np.hstack((weight_deltas.data, gradient.data))  # add the bias which is just the gradient * 1

            # update the weights
            # print("weight deltas", weight_deltas.data, weight_deltas.data.shape)
            # print("layer weights", self.layer_weights[i].data, self.layer_weights[i].data.shape)
            self.layer_weights[i] = self.layer_weights[i].add_matrix(weight_deltas)
            # update the bias weights
            self.layer_bias_weights[i] = self.layer_bias_weights[i].add_matrix(gradient)

            # hidden layer error matrix
            # print("weights shape", self.layer_weights[i].data.shape)
            # print("transposed weights shape", self.layer_weights[i].transpose().data.shape)
            # print("error shape", error_matrix.data.shape)
            error_matrix = self.layer_weights[i].transpose().multiply_matrix(error_matrix)
            # print("new error matrix", error_matrix.data, error_matrix.data.shape)
