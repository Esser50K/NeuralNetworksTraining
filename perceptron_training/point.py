class Point:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
        self.label = -1

    def set_label(self, label: int):
        self.label = label

    def __str__(self) -> str:
        return f'({self.x}, {self.y})'

class Line:
    def __init__(self, slope: float, intercept: float):
        self.slope = slope
        self.intercept = intercept

    def label(self, point: Point) -> int:
        return 1 if point.y > self.slope * point.x + self.intercept else -1

    def line_in_rect(self,
                     rect: tuple[int, int, int, int],
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        rx1 = rect[0]
        rx2 = rect[2]

        x1 = rx1
        x2 = rx2
        y1 = self.slope * x1 + self.intercept
        y2 = self.slope * x2 + self.intercept
        return (x1, y1), (x2, y2)

    def __str__(self) -> str:
        return f'y = {self.slope}x + {self.intercept}'

def label_point(point: Point, line: Line) -> int:
    return line.label(point)