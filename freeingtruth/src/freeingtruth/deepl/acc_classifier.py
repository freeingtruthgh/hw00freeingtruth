import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

class ResConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block= nn.Sequential(
            nn.Conv1d(in_channels,out_channels,kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
        )

        # Match dimensions for skip connection
        self.downsample = (
            nn.Sequential(
                nn.Conv1d(in_channels,out_channels,kernel_size=1),
                nn.BatchNorm1d(out_channels)
            )
            if in_channels != out_channels else nn.Identity()
        )
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self,x):
        identity = self.downsample(x)
        out = self.block(x)
        out += identity
        return self.relu(out)

class ACCNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Blocks 1-5
        self.block1 = nn.Sequential(
            nn.Conv1d(1,16,kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            )

        # Residual blocks
        self.blocks = nn.Sequential(
            ResConvLayer(16, 32),
            ResConvLayer(32, 64),
            ResConvLayer(64, 64),
        )

        self.AvgPool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.5) 
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.block1(x)
        x = self.blocks(x)
        x = self.flatten(self.AvgPool(x))
        x = self.relu(self.dropout(self.fc1(x)))
        x = self.fc2(x)
        return x

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)

        probs = probs.view(-1)
        targets = targets.view(-1).float()

        intersection = (probs * targets).sum()
        denominator = probs.sum() + targets.sum()

        dice_coeff = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return 1.0 - dice_coeff

class ACCTrainer:
    def __init__(self, train_loader, val_loader, eta, epoch, loss=None, optimizer=None, model=None, device=None):
        # Training data and hyperparameters
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.eta = eta
        self.epoch = epoch

         # Device setup to GPU else cpu
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        if model is None:    
            self.model = ACCNet()
        else:
            self.model = model

        # Loss function
        if loss is None:
            self.loss = DiceLoss()
        else:
            self.loss = loss
        
        # Optimizer
        if optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.eta, weight_decay = 1e-4)
        else:
            self.optimizer = optimizer

        # step learning rate scheduler
        self.lrscheduler = optim.lr_scheduler.StepLR(self.optimizer,step_size=30, gamma=0.1)

        # set after training, initialize as empty torch Tensors
        self.train_loss_vector = torch.zeros(self.epoch)
        self.Val_loss_vector = torch.zeros(self.epoch)
        self.train_acc_vector = torch.zeros(self.epoch)
        self.Val_acc_vector = torch.zeros(self.epoch)

    def train(self):
        best_val_acc = 0.0
        for ep in range(self.epoch):
            # Training phase
            self.model.train()
            epoch_loss = 0
            correct =0
            total = 0

            for batch_idx, batch in enumerate(self.train_loader):
                batch_X = batch["x"].to(self.device).float()
                batch_y = batch["y"].to(self.device).float()
            
                self.optimizer.zero_grad()
                #Forward pass
                outputs = self.model(batch_X)

                # Compute Loss then backward pass
                train_loss = self.loss(outputs, batch_y)    # might need to add .sum()
                train_loss.backward()
                self.optimizer.step()

                epoch_loss += train_loss.item()
                probs = torch.sigmoid(outputs)
                preds = (probs >= 0.5).float()
                correct += (preds == batch_y).sum().item()
                total += batch_y.numel()

                train_acc = correct/total

                # Print every 10 batches
                if (batch_idx +1) % 5000 == 0:
                    self.model.eval()
                    val_correct = 0
                    val_total = 0

                    with torch.no_grad():
                        for i, val_batch in enumerate(self.val_loader):
                            if i == 2:  # only 2 batches
                                break

                            val_X = val_batch["x"].to(self.device).float()
                            val_y = val_batch["y"].to(self.device).float()

                            val_outputs = self.model(val_X)
                            val_probs = torch.sigmoid(val_outputs)
                            val_preds = (val_probs >= 0.5).float()
                            val_correct += (val_preds == val_y).sum().item()
                            val_total += val_y.numel()

                    val_acc = val_correct/val_total
                    self.model.train()

                    print(f'Epoch {ep+1}, Batch {batch_idx+1} | Train Loss: {train_loss.item():.6f}, train acc: {train_acc:.6f}, Val acc: {val_acc:.6f}', flush = True)
                    if torch.cuda.is_available():
                        print(f'GPU Memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB', flush=True)



            self.train_loss_vector[ep] = epoch_loss / len(self.train_loader)
            self.train_acc_vector[ep] = correct/total

            # Validation phase
            self.model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for batch in self.val_loader:
                    batch_X = batch["x"].to(self.device).float()
                    batch_y = batch["y"].to(self.device).float()
                    
                    val_outputs = self.model(batch_X)
                    validation_loss = self.loss(val_outputs, batch_y)
                    val_loss += validation_loss.item()
                    val_probs = torch.sigmoid(val_outputs)
                    val_preds = (val_probs >= 0.5).float()
                    val_correct += (val_preds == batch_y).sum().item()
                    val_total += batch_y.numel()

            self.Val_loss_vector[ep] = val_loss / len(self.val_loader)
            self.Val_acc_vector[ep] = val_correct/val_total
            current_val_acc = self.Val_acc_vector[ep]

            # Save best model
            if current_val_acc > best_val_acc:
                best_val_acc = current_val_acc

                torch.save({
                    'epoch': ep+1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': best_val_acc,
                }, "best_ACCNet_model.pth")

                print(f"Saved best model at epoch {ep+1} (Val Acc: {best_val_acc:.6f})", flush=True)

            # step scheduler
            self.lrscheduler.step()
            
        return [self.train_loss_vector.cpu(), self.Val_loss_vector.cpu(), self.train_acc_vector.cpu(), self.Val_acc_vector.cpu()]

    def save(self, filename="Trained_ACCNet.onnx"):
        self.model.eval()

        batch = next(iter(self.train_loader))
        inputs = batch["x"]
        ex_inputs = inputs[0:1]
        model_cpu = self.model.to("cpu")
        ex_inputs = ex_inputs.to("cpu")

        with torch.no_grad():
            torch.onnx.export(
                model_cpu,ex_inputs,
                filename,
                input_names=["input"], 
                output_names=["output"], 
                dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}, 
                opset_version=18
            )
        print(f"Model exported to {filename}")

    def evaluation(self):
        # plot training loss vs epochs
        plt.figure()
        plt.plot(self.train_loss_vector.cpu(),linewidth=2, label="Training")
        plt.plot(self.Val_loss_vector.cpu(),linewidth=2, label="Validation")
        # create title, and x, y labels
        plt.title("Loss vs Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig("results/accnet_Loss_vs_epochs.pdf")    # save PDF of plot
        plt.close()

        # plot training accuracy vs epochs
        plt.figure()
        plt.plot(self.train_acc_vector.cpu(),linewidth=2, label="Training")
        plt.plot(self.Val_acc_vector.cpu(),linewidth=2, label="Validation")
        # create title, and x, y labels
        plt.title("Accuracy vs Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig("results/accnet_Accuracy_vs_Epochs.pdf")    # save PDF of plot
        plt.close()