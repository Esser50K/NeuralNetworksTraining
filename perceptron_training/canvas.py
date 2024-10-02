import time
from time import sleep

import pygame
import random
from threading import Thread

from pygame.display import update

from perceptron_training.perceptron import randomly_weighted_perceptron, statically_weighted_perceptron
from perceptron_training.point import Point, Line, label_point


def init_canvas(width, height) -> pygame.Surface:
    pygame.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Perceptron Training")
    # paint the whole screen white
    screen.fill((255, 255, 255))

    return screen

def wait_for_keypress():
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                return

def main():
    width = 500
    height = 500

    canvas = init_canvas(width, height)

    # create decision boundary
    decision_boundary = Line(
        1.5,
        random.uniform(-2, -1),
    )
    # decision_boundary = Line(.5, 1)
    line_in_rect = decision_boundary.line_in_rect((0, 0, width, height))

    # generate training and test points
    number_of_training_inputs = 1000  # with 2 more zeroes this does converge more
    number_of_test_points = 100
    training_points = [
        Point(random.randrange(0, width), random.randrange(0, height))
        for _ in range(number_of_training_inputs)
    ]
    for point in training_points:
        point.set_label(decision_boundary.label(point))
    test_points = [
        Point(random.randrange(0, width), random.randrange(0, height))
        for _ in range(number_of_test_points)
    ]
    for point in test_points:
        point.set_label(decision_boundary.label(point))

    for point in test_points:
        pygame.draw.circle(canvas,
                           (150, 150, 150),
                           (int(point.x), int(point.y)),
                           5)
    pygame.display.update()
    wait_for_keypress()

    # draw decision boundary
    pygame.draw.line(canvas, (0, 0, 255), line_in_rect[0], line_in_rect[1], 5)
    pygame.display.update()
    wait_for_keypress()

    # perceptron = randomly_weighted_perceptron(2, .00005)
    perceptron = statically_weighted_perceptron(.00005)
    print("initial perceptron weights:", perceptron.weights, perceptron.bias)
    print("initial perceptron line:", perceptron.decision_boundary())
    print("desired decision line:", decision_boundary)
    guessed_decision_boundary = perceptron.decision_boundary()
    guessed_line_in_rect = guessed_decision_boundary.line_in_rect((0, 0, width, height))
    pygame.draw.line(canvas, (0, 0, 0), guessed_line_in_rect[0], guessed_line_in_rect[1], 3)
    pygame.display.update()
    wait_for_keypress()

    for point in test_points:
        inputs = [point.x, point.y]
        guess = perceptron.guess(inputs)
        pygame.draw.circle(canvas,
                           (0, 255, 0) if point.label == guess else (255, 0, 0),
                           (int(point.x), int(point.y)),
                           5)
        pygame.display.update()
        time.sleep(0.02)

    pygame.display.update()
    wait_for_keypress()

    canvas.fill((255, 255, 255))  # reset canvas
    pygame.draw.line(canvas, (0, 0, 255), line_in_rect[0], line_in_rect[1], 5)

    for i, point in enumerate(training_points):
        inputs = [point.x, point.y]
        perceptron.train(inputs, point.label)

        # redraw canvas
        canvas.fill((255, 255, 255))  # reset canvas
        pygame.draw.line(canvas, (0, 0, 255), line_in_rect[0], line_in_rect[1], 5)

        for test_point in test_points:
            inputs = [test_point.x, test_point.y]
            guess = perceptron.guess(inputs)
            pygame.draw.circle(canvas,
                               (0, 255, 0) if test_point.label == guess else (255, 0, 0),
                               (int(test_point.x), int(test_point.y)),
                               5)

        guessed_decision_boundary = perceptron.decision_boundary()
        guessed_line_in_rect = guessed_decision_boundary.line_in_rect((0, 0, width, height))
        pygame.draw.line(canvas, (0, 0, 0), guessed_line_in_rect[0], guessed_line_in_rect[1], 3)

        pygame.display.update()
        sleep(.01)

    print("final perceptron weights:", perceptron.weights, perceptron.bias)
    print("final perceptron line:", perceptron.decision_boundary())

    canvas = init_canvas(width, height)  # reset canvas
    pygame.draw.line(canvas, (0, 0, 255), line_in_rect[0], line_in_rect[1], 5)

    for point in test_points:
        inputs = [point.x, point.y]
        guess = perceptron.guess(inputs)
        pygame.draw.circle(canvas,
                           (0, 255, 0) if point.label == guess else (255, 0, 0),
                           (int(point.x), int(point.y)),
                           5)

    guessed_decision_boundary = perceptron.decision_boundary()
    guessed_line_in_rect = guessed_decision_boundary.line_in_rect((0, 0, width, height))
    pygame.draw.line(canvas, (0, 0, 0), guessed_line_in_rect[0], guessed_line_in_rect[1], 3)

    pygame.display.update()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:  # Use ESC key to quit
                    running = False

    pygame.quit()

if __name__ == '__main__':
    main()