import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class SimpleNN(nn.Module):
    def __init__(self, num_classes, in_features):
        super(SimpleNN, self).__init__()
        self.num_classes = num_classes
        self.in_features = in_features
        self.fc1 = nn.Linear(self.in_features, 3)
        self.fc2 = nn.Linear(3, 4)
        self.fc3 = nn.Linear(4, 5)
        self.fc4 = nn.Linear(5, self.num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class ClassTrainer(nn.Module):
    def __init__(self, X_train, Y_train, eta, epoch, loss=None, optimizer=None, model=None, device=None):
        # Training data and hyperparameters
        self.X_train = X_train
        self.Y_train = Y_train
        self.eta = eta
        self.epoch = epoch
        
        # Device setup to GPU else cpu
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        if model is None:    
            self.model = SimpleNN(self.num_classes, self.in_features)
        else:
            self.model = model

        # Loss function
        if loss is None:
            self.loss = nn.CrossEntropyLoss()
        else:
            self.loss = loss
        
        # Optimizer
        if optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.eta)
        else:
            self.optimizer = optimizer(self.model.parameters(), lr=self.eta)

        # set after training, initialize as empty torch Tensors
        self.loss_vector = torch.zeros(self.epoch, device=self.device)
        self.accuracy_vector = torch.zeros(self.epoch, device=self.device)

    def train(self):
        for ep in range(self.epoch):
            self.optimizer.zero_grad()
            #Forward pass
            Y_train_pred = self.model(self.X_train)

            # Compute Loss then backward pass
            train_loss = self.loss(Y_train_pred, self.Y_train).sum()
            train_loss.backward()

            self.loss_vector[ep] = train_loss.item()
            self.train_preds = torch.argmax(Y_train_pred,dim=1)
            acc = (self.train_preds == self.Y_train).float().mean()
            self.accuracy_vector[ep] = acc.item()
            self.optimizer.step()
            if (ep + 1) % 100 == 0:
                print(f'Epoch {ep+1}/{self.epoch}, Loss: {train_loss.item()}')
                print(f'GPU Memory: {torch.cuda.memory_allocated() / 1024**2:.5f} MB')
            
        return [self.loss_vector.cpu(), self.accuracy_vector.cpu()]


    def test(self, X_test, Y_test):
        self.model.eval()
        self.Y_test = Y_test
        with torch.no_grad():
            Y_test_pred = self.model(X_test)
            test_loss = self.loss(Y_test_pred, self.Y_test)
            self.test_preds = torch.argmax(Y_test_pred,dim=1)
            test_acc = (self.test_preds == self.Y_test).float().mean()

        return [self.test_preds, test_acc.item()]

    def predict(self, x):
        self.model.eval()
        with torch.no_grad():
            Y_pred = self.model(x)

        return torch.argmax(Y_pred,dim=1).float()

    def save(self, filename="Trained_SimpleNN.onnx"):
        ex_inputs = torch.randn(1,self.X_train.size(1)).to(self.device)
        torch.onnx.export(self.model,ex_inputs,filename,input_names=["input"], output_names=["output"], opset_version=11)

    def evaluation(self):
        # plot training loss vs epochs
        plt.plot(self.loss_vector.cpu(),linewidth=2)
        # create title, and x, y labels
        plt.title("Training Loss vs Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Training Loss (Binary Cross Entropy Loss)")
        plt.show()

        # plot accuracy vs epochs
        plt.plot(self.accuracy_vector.cpu(),linewidth=2)
        # create title, and x, y labels
        plt.title("Training Accuracy vs Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Training Accuracy")
        plt.show()
        
        # Move Trained and tested labels to cpu and their predictions
        Y_train_np = self.Y_train.detach().cpu()
        train_preds_np = self.train_preds.detach().cpu()
        Y_test_np = self.Y_test.detach().cpu()
        test_preds_np = self.test_preds.detach().cpu()
        
        # Create Confusion Matrix
        cm = confusion_matrix(Y_train_np, train_preds_np)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.show()

        # Calucalte final accuracy, F1 score, precision and recall fortrain and test data
        train_f1 = f1_score(Y_train_np, train_preds_np, average='macro')
        train_precision = precision_score(Y_train_np, train_preds_np, average='macro',zero_division=0)
        train_recall = recall_score(Y_train_np, train_preds_np, average='macro')
        train_accuracy = accuracy_score(Y_train_np, train_preds_np)

        test_f1 = f1_score(Y_test_np, test_preds_np, average='macro')
        test_precision = precision_score(Y_test_np, test_preds_np, average='macro',zero_division=0)
        test_recall = recall_score(Y_test_np, test_preds_np, average='macro')
        test_accuracy = accuracy_score(Y_test_np, test_preds_np)
        return [train_f1, train_precision, train_recall, train_accuracy, test_f1, test_precision, test_recall, test_accuracy]

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block= nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self,x):
        return self.block(x)

class ImageNetCNN(nn.Module):
    def __init__(self,num_classes):
        super().__init__()

        # Blocks 1-5
        self.blocks = nn.Sequential(
            ConvLayer(3,64),
            ConvLayer(64,128),
            ConvLayer(128,256),
            ConvLayer(256,512),
            ConvLayer(512,512),
            )

        self.AvgPool = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512,1024)
        self.fc2 = nn.Linear(1024,num_classes)
        self.dropout = nn.Dropout(0.5) 
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.blocks(x)
        x = self.flatten(self.AvgPool(x))
        x = self.relu(self.dropout(self.fc1(x)))
        x = self.fc2(x)
        return x

class CNNTrainer(nn.Module):
    def __init__(self, train_loader, val_loader, eta, epoch, loss=None, optimizer=None, model=None, device=None):
        super().__init__()
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
            self.model = ImageNetCNN(self.num_classes)
        else:
            self.model = model

        # Loss function
        if loss is None:
            self.loss = nn.CrossEntropyLoss()
        else:
            self.loss = loss
        
        # Optimizer
        if optimizer is None:
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.eta, momentum = 0.9, weight_decay = 1e-4)
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
                batch_X = batch["pixel_values"].to(self.device)
                batch_y = batch["labels"].to(self.device)
            
                self.optimizer.zero_grad()
                #Forward pass
                outputs = self.model(batch_X)

                # Compute Loss then backward pass
                train_loss = self.loss(outputs, batch_y)    # might need to add .sum()
                train_loss.backward()
                self.optimizer.step()

                epoch_loss += train_loss.item()
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == batch_y).sum().item()
                total += batch_y.size(0)

                train_acc = correct/total

                # Print every 10 batches
                if (batch_idx +1) % 10 == 0:
                    self.model.eval()
                    val_correct = 0
                    val_total = 0

                    with torch.no_grad():
                        for i, val_batch in enumerate(self.val_loader):
                            if i == 2:  # only 2 batches
                                break

                            val_X = val_batch["pixel_values"].to(self.device)
                            val_y = val_batch["labels"].to(self.device)

                            val_outputs = self.model(val_X)
                            val_preds = torch.argmax(val_outputs, dim=1)
                            val_correct += (val_preds == val_y).sum().item()
                            val_total += val_y.size(0)

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
                    batch_X = batch["pixel_values"].to(self.device)
                    batch_y = batch["labels"].to(self.device)
                    
                    outputs = self.model(batch_X)
                    validation_loss = self.loss(outputs, batch_y)
                    val_loss += validation_loss.item()
                    preds = torch.argmax(outputs, dim=1)
                    val_correct += (preds == batch_y).sum().item()
                    val_total += batch_y.size(0)

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
                }, "best_CNN_model.pth")

                print(f"Saved best model at epoch {ep+1} (Val Acc: {best_val_acc:.6f})", flush=True)

            # step scheduler
            self.lrscheduler.step()

            if (ep + 1) % 5 == 0:
                torch.save({
                    'epoch': ep + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': self.train_loss_vector[ep].item(),
                    'val_loss': self.Val_loss_vector[ep].item(),
                    'train_acc': self.train_acc_vector[ep].item(),
                    'val_acc': self.Val_acc_vector[ep].item(),
                }, f"checkpoint_epoch_{ep+1}.pth")

                print(f"Saved checkpoint at epoch {ep+1}", flush=True)
            
        return [self.train_loss_vector.cpu(), self.Val_loss_vector.cpu(), self.train_acc_vector.cpu(), self.Val_acc_vector.cpu()]

    def save(self, filename="Trained_ImagenetCNN.onnx"):
        self.model.eval()

        batch = next(iter(self.train_loader))
        inputs = batch["pixel_values"]
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
        plt.ylabel("Loss (Cross Entropy Loss)")
        plt.legend()
        plt.savefig("results/hw03_Loss_vs_epochs.pdf")    # save PDF of plot
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
        plt.savefig("results/hw03_Accuracy_vs_Epochs.pdf")    # save PDF of plot
        plt.close()

