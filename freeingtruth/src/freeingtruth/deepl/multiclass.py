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

class ClassTrainer:
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
            train_loss = self.loss(Y_train_pred, self.Y_train)
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
