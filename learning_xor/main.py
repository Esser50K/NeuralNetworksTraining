import pygame
from neural_network.neuralnet import NeuralNetwork

training_data = {
    (0, 0): 0,
    (0, 1): 1,
    (1, 0): 1,
    (1, 1): 0
}

def init_canvas(width, height) -> pygame.Surface:
    pygame.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Perceptron Training")
    # paint the whole screen black
    screen.fill((0, 0, 0))

    return screen

def main():
    nn = NeuralNetwork([2, 10, 10, 1], learning_rate=0.1)

    width = 500
    height = 500

    screen = init_canvas(width, height)
    resolution = 10
    cols = width // resolution
    rows = height // resolution

    running = True
    while running:
        for i in range(cols):
            for j in range(rows):
                inputs = [i / cols, j / rows]
                guess = nn.forward(inputs)[0]
                pygame.draw.rect(screen, (int(255*guess), int(255*guess), int(255*guess)), (i * resolution, j * resolution, resolution - 1, resolution - 1))

        pygame.display.update()

        for epoch in range(100):
            for inputs, target in training_data.items():
                nn.train(list(inputs), [target])

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

    pygame.quit()

if __name__ == '__main__':
    main()