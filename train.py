import os
import sys
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from data_loader import ChessDataset

class HumanNNUE(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu_stack = nn.Sequential(
            nn.Linear(2 * 64 * 64 * 10 + 64 * 64, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 64 * 64),
        )

    def forward(self, x):
        logits = self.relu_stack(x)
        return logits
    
    def loss_fn(self):
        return nn.CrossEntropyLoss() #(pred[:3], actual[:3]) + nn.CrossEntropyLoss()(pred[3:], actual[3:])

def train_loop(dataloader, model, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = model.loss_fn()(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += model.loss_fn()(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.2f}%, Avg loss: {test_loss:>8f} \n")

if __name__ == "__main__":
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device: {torch.cuda.get_device_name() if device == 'cuda' else 'CPU'}")

    learning_rate = 0.2e-3
    batch_size = 32
    epochs = 100

    if len(sys.argv) > 1:
        print(f"Loading model {sys.argv[1]}")
        model = torch.load(sys.argv[1])
        start_t = int(sys.argv[1].split('_')[-1].split('.')[0]) + 1
    else:
        model = HumanNNUE()
        start_t = 0

    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # Load dataset
    print("Loading data")
    train_dataloader = DataLoader(ChessDataset('train.games', device), batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(ChessDataset('test.games', device), batch_size=batch_size, shuffle=True)

    print("Baseline:")
    test_loop(test_dataloader, model)
    for t in range(start_t, start_t + epochs):
        print(f"Epoch {t}\n-------------------------------")
        train_loop(train_dataloader, model, optimizer)
        test_loop(test_dataloader, model)
        torch.save(model, f'human_nnue_{t}.pth')
    print("Done!")