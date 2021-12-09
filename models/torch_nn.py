import util
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torch.utils.data as data_utils
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from imbalanced import ImbalancedDatasetSampler


# ref: https://www.kaggle.com/dakshmiglani/pytorch-credit-card-fraud-prediction-99-8

class FraudNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(29, 16)
        self.bn1 = nn.BatchNorm1d(16)
        self.fc2 = nn.Linear(16, 18)
        self.bn2 = nn.BatchNorm1d(18)
        self.fc3 = nn.Linear(18, 20)
        self.bn3 = nn.BatchNorm1d(20)
        self.fc4 = nn.Linear(20, 24)
        self.fc5 = nn.Linear(24, 1)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = self.bn1(x)
        x = F.leaky_relu(self.fc2(x))
        x = self.bn2(x)
        x = F.leaky_relu(self.fc3(x))
        x = self.bn3(x)
        x = self.fc4(x)
        x = torch.sigmoid(self.fc5(x))
        return x


def main():
    torch.manual_seed(2333)

    # prepare dataset
    opts = util.parse_args()
    X, y = util.data_load(opts.dataset)
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train, X_test = util.normalize(X_train, X_test)

    print(f"Before upsample: {len(X_train)}, we have {np.sum(Y_train)} minority data")
    # upsample
    diff = len(Y_train) - np.sum(Y_train)
    # X_train, Y_train = util.upsample(X_train, Y_train, 2048)
    X_minority = util.extract_true(X_train, Y_train)

    needed = diff / len(X_minority)
    needed = int(needed)
    needed = 10
    X_train = np.concatenate((X_train, *[X_minority for _ in range(needed)]), axis=0)
    Y_train = np.concatenate((Y_train, *[[1] for _ in range(needed * len(X_minority))]), axis=0)
    assert(len(X_train) == len(Y_train))

    print(f"train on size: {len(X_train)}, we have {np.sum(Y_train)} minority data")

    X_train = torch.from_numpy(X_train)
    Y_train = torch.from_numpy(Y_train)

    X_test = torch.from_numpy(X_test)
    Y_test = torch.from_numpy(Y_test)

    model = FraudNet().float()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.02)

    training_epochs = 2
    minibatch_size = 2 << 10

    train_dataset = data_utils.TensorDataset(X_train, Y_train)
    train_loader = data_utils.DataLoader(train_dataset, sampler=ImbalancedDatasetSampler(train_dataset),
                                         batch_size=minibatch_size)
    # train_loader = data_utils.DataLoader(train_dataset,
    #                                      batch_size=minibatch_size, shuffle=True)

    test_dataset = data_utils.TensorDataset(X_test, Y_test)
    test_loader = data_utils.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True)

    label_name = ["normal", "fraud"]
    for epoch in range(training_epochs):
        model.train()
        for b, data in enumerate(train_loader):
            inputs, labels = data
            y_pred = model(inputs).float()
            labels = torch.unsqueeze(labels, 1).float()
            loss = criterion(y_pred, labels)

            # if b % 256:
            #     print('Epochs: {}, batch: {} loss: {}'.format(epoch, b, loss))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            for data in test_loader:
                inputs, labels = data
                labels = torch.unsqueeze(labels, 1).float()

                outputs = model(inputs.float())
                loss = criterion(outputs, labels)
                print(f"test loss: {loss}")

                # print(f"fraud output is: {outputs[labels == 1.0]}")
                print(f"minimal output is {torch.min(outputs)}")

                outputs_binary = (outputs > 0.95)
                print(outputs_binary)
                print(f"mean output: {torch.mean(outputs)}")
                cm = confusion_matrix(labels, outputs_binary)
                test_result = classification_report(labels, outputs_binary, target_names=label_name)
                print(f"confusion matrix: {cm}")
                print(test_result)


if __name__ == '__main__':
    main()
