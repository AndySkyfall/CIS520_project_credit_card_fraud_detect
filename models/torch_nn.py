import matplotlib.pyplot as plt
import util
import sys
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torch.utils.data as data_utils
from sklearn.model_selection import train_test_split


# ref: https://www.kaggle.com/dakshmiglani/pytorch-credit-card-fraud-prediction-99-8

class FraudNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(30, 16)
        self.fc2 = nn.Linear(16, 18)
        self.fc3 = nn.Linear(18, 20)
        self.fc4 = nn.Linear(20, 24)
        self.fc5 = nn.Linear(24, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.25)
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.sigmoid(self.fc5(x))
        return x


def main():
    torch.manual_seed(2333)

    # prepare dataset
    opts = util.parse_args()
    X, y = util.data_load(opts.dataset)
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train, X_test = util.normalize(X_train, X_test)

    print(f"Before upsample: {len(X_train)}, we have {np.sum(Y_train)} fraud train size")
    # upsample
    X_train, Y_train = util.upsample(X_train, Y_train, 2)

    print(f"train on size: {len(X_train)}, we have {np.sum(Y_train)} fraud train size")

    X_train = torch.from_numpy(X_train)
    Y_train = torch.from_numpy(Y_train).double()

    model = FraudNet()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    training_epochs = 2
    minibatch_size = 64

    train_dataset = data_utils.TensorDataset(X_train, Y_train)
    train_loader = data_utils.DataLoader(train_dataset, batch_size=minibatch_size, shuffle=True)

    for i in range(training_epochs):
        for b, data in enumerate(train_loader, 0):
            inputs, labels = data
            y_pred = model(inputs)
            loss = criterion(y_pred, labels)

            if b % 100:
                print('Epochs: {}, batch: {} loss: {}'.format(i, b, loss))
            # reset gradients
            optimizer.zero_grad()

            # backward pass
            loss.backward()

            # update weights
            optimizer.step()


if __name__ == '__main__':
    main()
