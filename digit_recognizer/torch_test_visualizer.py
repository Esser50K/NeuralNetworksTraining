import pygame
import torch
import numpy as np

from torch_neural_network.cnn import CNN
from PIL import Image

def gaussian_kernel(size, sigma=1):
    """Generates a Gaussian kernel."""
    kernel = np.fromfunction(
        lambda x, y: (1/ (2 * np.pi * sigma**2)) * np.exp(-((x - (size - 1) / 2) ** 2 + (y - (size - 1) / 2) ** 2) / (2 * sigma ** 2)),
        (size, size)
    )
    return kernel / np.sum(kernel)  # Normalize the kernel

def convolve2d(image, kernel):
    """Performs 2D convolution."""
    kernel_size = kernel.shape[0]
    pad = kernel_size // 2
    padded_image = np.pad(image, pad, mode='constant')
    result = np.zeros_like(image, dtype=float)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            result[i, j] = np.sum(padded_image[i:i + kernel_size, j:j + kernel_size] * kernel)

    return result

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

    n_outputs = 10
    cnn = CNN()
    cnn.load_state_dict(torch.load("digit_recognizer/torch_weights/5_epoch", weights_only=False))
    cnn.eval()

    # draw neural network output
    screen, drawing_canvas_dimensions, neuralnet_output_canvas_dimensions = create_canvas(500)
    cols = 28
    rows = 28

    # gaussian blur kernel to apply to drawn image
    kernel = gaussian_kernel(size=3, sigma=1)
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
                    pygame.draw.circle(screen, (255, 255, 255, 0.2), (mouse_x, mouse_y), 15)
                    pygame.display.update()

                # reduce drawing canvas to neural network input size
                inputs = []
                for i in range(rows):
                    for j in range(cols):
                        x = int(j * (drawing_canvas_dimensions[0] / cols))
                        y = int(i * (drawing_canvas_dimensions[1] / rows))

                        input_value = screen.get_at((x, y))[0] / 255
                        inputs.append(input_value)

                img_inputs = np.array(inputs).reshape(28, 28)
                blurred_image = convolve2d(img_inputs.astype(np.float32), kernel)
                blurred_image = blurred_image.reshape(1, 1, 28, 28)
                tensor = torch.from_numpy(blurred_image).float()
                with torch.no_grad():
                    output = cnn(tensor)

                probabilities = torch.softmax(output, dim=1)
                guessed_digit = torch.argmax(probabilities, dim=1).item()

                print(f"guessed digit: {guessed_digit}")
                print("confidence:", probabilities[0][guessed_digit])
                print("probabilities:", probabilities[0])

                # draw neural network output

                drawing_canvas_y_offset = drawing_canvas_dimensions[1]
                full_canvas_height = drawing_canvas_dimensions[1] + neuralnet_output_canvas_dimensions[1]
                y_center_of_output_canvas = drawing_canvas_y_offset + (neuralnet_output_canvas_dimensions[1] / 2)
                x_center_of_output_canvas_offset = neuralnet_output_canvas_dimensions[0] / n_outputs / 2
                rect_width = neuralnet_output_canvas_dimensions[0] / n_outputs
                # clear the neural network output canvas
                pygame.draw.rect(screen, (0, 0, 0), (0, drawing_canvas_y_offset, neuralnet_output_canvas_dimensions[0], full_canvas_height))

                for i in range(n_outputs):
                    rect_height = probabilities[0][i].item() * neuralnet_output_canvas_dimensions[1]
                    rect_y_start = full_canvas_height - rect_height
                    rect_x_start = i * (neuralnet_output_canvas_dimensions[0] / n_outputs)
                    pygame.draw.rect(
                        screen,
                        (255, 255, 255),
                        (rect_x_start, rect_y_start, rect_width, probabilities[0][i].item() * neuralnet_output_canvas_dimensions[1]))

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
                saveable_inputs = np.array(inputs).reshape(28, 28)
                blurred_image = convolve2d(saveable_inputs.astype(np.float32), kernel)
                blurred_image = blurred_image * 255

                image = Image.fromarray(blurred_image.astype(np.uint8), mode='L')
                image.save("input.png")


if __name__ == '__main__':
    main()