import pygame
import numpy as np
from neural_network.neuralnet import NeuralNetwork
from PIL import Image

def create_canvas(width) -> (pygame.Surface, tuple, tuple):
    drawing_canvas_dimensions = (width, width)
    neuralnet_output_canvas_dimensions = (width, width / 10)

    screen = pygame.display.set_mode((width, width + neuralnet_output_canvas_dimensions[1]))
    pygame.display.set_caption("Digit Recognizer")
    # paint the whole screen black
    screen.fill((0, 0, 0))
    # draw separation line between canvases
    pygame.draw.line(screen, (255, 255, 255), (0, width), (width, width), 5)
    pygame.display.update()

    return screen, drawing_canvas_dimensions, neuralnet_output_canvas_dimensions

def main():
    pygame.init()

    n_inputs = 28 * 28
    n_outputs = 10
    nn = NeuralNetwork([n_inputs, 1000, 100, 50, n_outputs])
    nn.load_weights("digit_recognizer/weights/1_epoch")

    # draw neural network output
    screen, drawing_canvas_dimensions, neuralnet_output_canvas_dimensions = create_canvas(500)
    cols = 28
    rows = 28

    running = True
    is_drawing = False
    inputs = []
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.MOUSEBUTTONDOWN:
                is_drawing = True

            if is_drawing and event.type == pygame.MOUSEBUTTONUP:
                is_drawing = False

            if is_drawing:
                mouse_pos = pygame.mouse.get_pos()
                mouse_x = mouse_pos[0]
                mouse_y = mouse_pos[1]
                if mouse_x < drawing_canvas_dimensions[0] and mouse_y < drawing_canvas_dimensions[1]:
                    # draw on canvas
                    pygame.draw.circle(screen, (255, 255, 255), (mouse_x, mouse_y), 20)
                    pygame.display.update()

                # reduce drawing canvas to neural network input size
                inputs = []
                for i in range(cols):
                    for j in range(rows):
                        x = int(i * (drawing_canvas_dimensions[0] / rows))
                        y = int(j * (drawing_canvas_dimensions[1] / cols))

                        input_value = screen.get_at((y, x))[0] / 255
                        inputs.append(input_value)

                # print inputs grid as ascii art
                ascii_input = ""
                for i in range(cols):
                    for j in range(rows):
                        idx = (i * rows) + j
                        if inputs[idx] > 0.5:
                            ascii_input += "#"
                        else:
                            ascii_input += " "
                    ascii_input += "\n"
                print(ascii_input)

                output = nn.forward(inputs)
                guessed_digit = output.index(max(output))
                print(f"guessed digit: {guessed_digit}")
                print("outputs:", output)

                # draw neural network output

                drawing_canvas_y_offset = drawing_canvas_dimensions[1]
                full_canvas_height = drawing_canvas_dimensions[1] + neuralnet_output_canvas_dimensions[1]
                y_center_of_output_canvas = drawing_canvas_y_offset + (neuralnet_output_canvas_dimensions[1] / 2)
                x_center_of_output_canvas_offset = neuralnet_output_canvas_dimensions[0] / n_outputs / 2
                rect_width = neuralnet_output_canvas_dimensions[0] / n_outputs
                # clear the neural network output canvas
                pygame.draw.rect(screen, (0, 0, 0), (0, drawing_canvas_y_offset, neuralnet_output_canvas_dimensions[0], full_canvas_height))

                for i in range(n_outputs):
                    rect_height = output[i] * neuralnet_output_canvas_dimensions[1]
                    rect_y_start = full_canvas_height - rect_height
                    rect_x_start = i * (neuralnet_output_canvas_dimensions[0] / n_outputs)
                    pygame.draw.rect(
                        screen,
                        (255, 255, 255),
                        (rect_x_start, rect_y_start, rect_width, output[i] * neuralnet_output_canvas_dimensions[1]))

                # draw output output labels
                font = pygame.font.Font(None, 30)
                for i in range(n_outputs):
                    text = font.render(str(i), True, (255, 0, 0))  # White color text
                    screen.blit(text, (i * (neuralnet_output_canvas_dimensions[0] / n_outputs) + x_center_of_output_canvas_offset, y_center_of_output_canvas))

                pygame.display.update()

            # clear canvas when pressing r
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                inputs = []
                screen.fill((0, 0, 0))
                pygame.draw.line(screen, (255, 255, 255), (0, drawing_canvas_dimensions[1]), (drawing_canvas_dimensions[0], drawing_canvas_dimensions[1]), 5)
                pygame.display.update()

            # save input as 28x28 image when pressing s
            if event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                print("Saving input as image")
                # make inputs be 255 instead of 1
                saveable_inputs = np.array(inputs) * 255

                image = Image.fromarray(saveable_inputs.reshape(28, 28).astype(np.uint8), mode='L')
                image.save("input.png")


if __name__ == '__main__':
    main()