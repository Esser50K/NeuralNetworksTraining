import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch_neural_network.cnn import CNN
from mnist_decoder import MNISTDecoder

class MNISTDataset(Dataset):
    def __init__(self, images_path: str, labels_path: str):
        self.decoder = MNISTDecoder(images_path, labels_path)
        self.decoder.init()  # Initialize the decoder
        self.n_items = self.decoder.n_items

    def __len__(self):
        return self.n_items

    def __getitem__(self, idx):
        image, label = self.decoder.get_image_and_label_at(idx)
        image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # Shape: (1, 28, 28)
        image_tensor /= 255.0  # Normalize to [0, 1]
        return image_tensor, label

def train(model, device, train_loader, optimizer, criterion, epoch):
    # Put model in training mode. This will "enable" the dropout layer.
    # It only runs in training mode, not in evaluation mode.
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()  # Reset the gradients
        output = model(data)  # Forward pass
        loss = criterion(output, target)  # Calculate the loss or error
        loss.backward()  # backpropagation
        optimizer.step()  # Update the weights
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}]\tLoss: {loss.item():.6f}')

def main():
    # Create the dataset and dataloader
    mnist_data = "mnist_dataset"
    mnist_dataset = MNISTDataset(
        f"{mnist_data}/train-images-idx3-ubyte",
        f"{mnist_data}/train-labels-idx1-ubyte"
    )
    train_loader = DataLoader(mnist_dataset, batch_size=64, shuffle=True)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Instantiate the model, define the loss function and optimizer
    cnn = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn.parameters(), lr=0.001)

    # Training loop
    num_epochs = 5
    for epoch in range(num_epochs):
        train(cnn, device, train_loader, optimizer, criterion, epoch)

    print("Training completed, saving weights...")
    torch.save(cnn.state_dict(), f"digit_recognizer/torch_weights/{num_epochs}_epoch")

if __name__ == '__main__':
    main()