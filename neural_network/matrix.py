import numpy as np

class Matrix:
    def __init__(self, rows: int, cols: int, randomize: bool = False):
        self.rows = rows
        self.cols = cols
        if randomize:
            self.data = np.random.uniform(-1, 1, (rows, cols))
        else:
            self.data = np.matrix(np.zeros((rows, cols)))

    def __str__(self) -> str:
        return str(self.data)

    def randomize(self):
        self.data = np.random.uniform(-1, 1, (self.rows, self.cols))

    def multiply_scalar(self, scalar: float) -> 'Matrix':
        data = self.data * scalar
        out = Matrix(data.shape[0], data.shape[1])
        out.data = data

        return out

    def add_scalar(self, scalar: float):
        data = self.data + scalar
        out = Matrix(data.shape[0], data.shape[1])
        out.data = data

        return out

    def add_matrix(self, other: 'Matrix'):
        data = self.data + other.data
        out = Matrix(data.shape[0], data.shape[1])
        out.data = data

        return out

    def subtract_matrix(self, other: 'Matrix'):
        data = self.data - other.data
        out = Matrix(data.shape[0], data.shape[1])
        out.data = data

        return out

    def transpose(self):
        out = Matrix(self.cols, self.rows)
        out.data = self.data.T

        return out

    # some rules of matrix multiplication
    # matrices can only be multiplied if the number of columns in the first matrix is equal to the number of rows in the second
    # so Matrix(2, 3) * Matrix(3, 2) works but Matrix(2, 3) * Matrix(2, 3) doesn't
    # the result of a matrix multiplication is a new matrix with the number of rows of the first matrix and the number of columns of the second
    # so Matrix(2, 3) * Matrix(3, 2) = Matrix(2, 2)
    def multiply_matrix(self, other: 'Matrix'):
        data = self.data @ other.data
        rows, cols = self.data.shape

        out = Matrix(rows, cols)
        out.data = data

        return out

    def element_wise_multiply(self, other: 'Matrix'):
        data = np.multiply(self.data, other.data)
        out = Matrix(data.shape[0], data.shape[1])
        out.data = data

        return out

    def apply_fn(self, fn):
        data = np.vectorize(fn)(self.data)

        out = Matrix(data.shape[0], data.shape[1])
        out.data = data

        return out
