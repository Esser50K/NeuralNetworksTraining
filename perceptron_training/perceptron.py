import random

from perceptron_training.point import Line

class Perceptron:
    def __init__(self, weights: list[float], bias: float, learning_rate: float):
        self.weights = weights
        self.bias = bias
        self.learning_rate = learning_rate

    def guess(self, inputs: list[float]) -> int:
        weighted_sum = 0
        for i in range(len(inputs)):
            weighted_sum += self.weights[i] * inputs[i]
        return 1 if weighted_sum + self.bias > 0 else -1

    def decision_boundary(self) -> Line:
        # assuming there are only 2 inputs
        w1 = self.weights[0]
        w2 = self.weights[1]

        # The line equation is w1*x + w2*y + b = 0
        # Rearranging to slope-intercept form: y = (-w1/w2)x - (b/w2)
        slope = -w1 / w2
        intercept = -self.bias / w2

        return Line(slope, intercept)

    def train(self, inputs: list[float], label: int):
        guess = self.guess(inputs)
        error = label - guess
        for i in range(len(inputs)):
            self.weights[i] += error * inputs[i] * self.learning_rate

        self.bias += error * self.learning_rate

def randomly_weighted_perceptron(num_inputs: int, learning_rate: float) -> Perceptron:
    weights = [round(random.uniform(-10, 10), 2) for _ in range(num_inputs)]
    bias = round(random.uniform(-10, 10), 2)
    return Perceptron(weights, bias, learning_rate)

def statically_weighted_perceptron(learning_rate: float) -> Perceptron:
    weights = [-1, 3]
    bias = 2
    return Perceptron(weights, bias, learning_rate)
