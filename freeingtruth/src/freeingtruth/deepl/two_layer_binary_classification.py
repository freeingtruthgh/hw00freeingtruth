# /// script
# requires-python = ">=3.12"
# dependencies = [
#       "torch",
#       "numpy",
# ]
# ///

import torch
import numpy as np

def binary_classification(d: int,n: int, epochs: int=10000, lr: float=0.001) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[float]]:
    """
    Two layer binary classifications:
    This function trains a feedforward Nerual Network with two hidden layers and one output layer.
    The feature matrix will be gererated randomly. Each weight matrix is initialized from a Guassian
    Distribution with 0 mean and standard deviation sqrt(2/i) where 'i' is the number of inputs to the corresponding layer.
    A sigmoid activation function is applied after each layer and the network is trained using cross entropy loss.

    Args:
        d (int): number of features
        n (int): number of data samples
        epochs (int): Number of epochs for training (default is 10000)
        lr (float): Learning rate (default is 0.001)

    Returns:
        w_1 (torch.Tensor): Final updated weight matrix for input layer, size d,48
        w_2 (torch.Tensor): Final updated weight matrix for first hidden layer, size 48,16
        w_3 (torch.Tensor): Final updated weight matrix for second hidden layer, size 16,32
        w_4 (torch.Tensor): Final updated weight matrix for output layer, size 32,1
        loss (list[float]): History of the training loss value at every epoch
    """

    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    if device.type == 'cuda':
        print(f'GPU Name: {torch.cuda.get_device_name(0)}')
        print(f'GPU Capability: {torch.cuda.get_device_capability(0)}')

    # Create random features and labels
    torch.manual_seed(42)  # For reproducibility
    X = torch.randn(n, d, dtype=torch.float32, device=device)
    # Create labels by suming features for one data sample
    row_sum =torch.sum(X, dim=1, keepdim=True)
    # if greater than 2 the label is 1, otherwise 0
    Y= (row_sum>2).float()

    # Initialize Weights; setting requires_grad=True tracks the computational graph
    w_1 = torch.nn.Parameter(torch.randn(d, 48, dtype=torch.float32, device=device)* np.sqrt(2/d))
    w_2 = torch.nn.Parameter(torch.randn(48, 16, dtype=torch.float32, device=device)*np.sqrt(2/48))
    w_3 = torch.nn.Parameter(torch.randn(16, 32, dtype=torch.float32, device=device)*np.sqrt(2/16))
    w_4 = torch.nn.Parameter(torch.randn(32, 1, dtype=torch.float32, device=device)* np.sqrt(2/32))

    # lists to store weights and losses over epochs
    loss_vals = []

    # Define the loss function: 
    loss_fn = torch.nn.BCELoss(reduction='sum')

    for epoch in range(epochs):
        # Forward pass: Comput predicted y
        Z_1 = X@w_1
        A_1 = torch.sigmoid(Z_1@w_2)
        Z_2 = A_1@w_3
        A_3= torch.sigmoid(Z_2@w_4)
        Y_pred=A_3

        # Compute loss
        loss = loss_fn(Y_pred, Y)
        # Backward pass
        loss.backward()

        # Update parameters
        with torch.no_grad():
            w_1 -= lr*w_1.grad
            w_2 -= lr*w_2.grad
            w_3 -= lr*w_3.grad
            w_4 -= lr*w_4.grad

            # Zero the gradients
            w_1.grad.zero_()
            w_2.grad.zero_()
            w_3.grad.zero_()
            w_4.grad.zero_()

        # Store weight values at each epoch
        # move to CPU for storage in Python lists
        loss_vals.append(loss.item())

    return w_1.detach().cpu(), w_2.detach().cpu(), w_3.detach().cpu(), w_4.detach().cpu(), loss_vals