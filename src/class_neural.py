import torch
from scipy import sparse
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import logging
import tqdm
import numpy as np

torch.manual_seed(123321)
np.random.seed(123321)

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logging.getLogger(__name__).setLevel(logging.INFO)


class E2EDatasetLoader(Dataset):
    def __init__(self, features, targets=None):

        if "sparse" in str(type(features)):
            self.features = features.tocsr()
        else:
            self.features = sparse.csr_matrix(features)

        if targets is not None:
            self.targets = targets  # .tocsr()
        else:
            self.targets = targets

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, index):
        instance = torch.from_numpy(self.features[index, :].todense())

        if self.targets is not None:
            target = torch.from_numpy(np.array(self.targets[index]))
        else:
            target = None

        if target is None:
            return instance
        else:
            return instance, target


class AUTOENCODER(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 nn_type="mini",
                 dropout=0.1,
                 device="cuda"):
        super(AUTOENCODER, self).__init__()

        self.nn_type = nn_type
        if nn_type == "mini":
            self.hidden_first = nn.Linear(input_size, hidden_size)
            self.bn1 = nn.BatchNorm1d(hidden_size)
            self.hidden_second = nn.Linear(hidden_size, input_size)
            self.dropout = nn.Dropout(dropout)
            self.activation = nn.Softsign()

        elif nn_type == "large":
            self.hidden_first = nn.Linear(input_size, hidden_size)
            self.bn1 = nn.BatchNorm1d(hidden_size)
            self.hidden_first_v1 = nn.Linear(hidden_size, hidden_size)
            self.bn2 = nn.BatchNorm1d(hidden_size)
            self.hidden_second_v1 = nn.Linear(hidden_size, hidden_size)
            self.hidden_second = nn.Linear(hidden_size, input_size)
            self.dropout = nn.Dropout(dropout)
            self.activation = nn.Softsign()

    def forward(self, x):

        if self.nn_type == "mini":
            # attend and aggregate
            out = self.hidden_first(x)
            out = self.dropout(out)
            d1 = out.shape[0]
            out = out.view(d1, -1)
            out = self.bn1(out)
            out = self.activation(out)
            out = self.hidden_second(out)

        elif self.nn_type == "large":
            # attend and aggregate
            out = self.hidden_first(x)
            out = self.dropout(out)
            d1 = out.shape[0]
            out = out.view(d1, -1)
            out = self.bn1(out)
            out = self.activation(out)
            out = self.hidden_first_v1(out)
            out = self.dropout(out)
            d1 = out.shape[0]
            out = out.view(d1, -1)
            out = self.bn2(out)
            out = self.activation(out)
            out = self.hidden_second_v1(out)
            out = self.dropout(out)
            out = self.activation(out)
            out = self.hidden_second(out)

        return out

    def get_rep(self, x):

        if self.nn_type == "mini":
            out = self.hidden_first(x)

        elif self.nn_type == "large":
            out = self.hidden_first(x)
            out = self.dropout(out)
            out = self.bn1(out)
            out = self.activation(out)
            out = self.hidden_first_v1(out)

        return out


class GenericAutoencoder:
    def __init__(self,
                 batch_size=32,
                 num_epochs=99999,
                 learning_rate=0.001,
                 stopping_crit=30,
                 n_components=64,
                 nn_type="mini",
                 dropout=0.2,
                 verbose=False):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.loss = torch.nn.SmoothL1Loss()
        self.dropout = dropout
        self.nn_type = nn_type
        self.verbose = verbose
        self.batch_size = batch_size
        self.stopping_crit = stopping_crit
        self.num_epochs = num_epochs
        self.hidden_layer_size = n_components
        self.learning_rate = learning_rate
        self.model = None
        self.optimizer = None
        self.num_params = None

    def fit(self, features):

        train_dataset = E2EDatasetLoader(features)
        dataloader = DataLoader(train_dataset,
                                batch_size=self.batch_size,
                                shuffle=True,
                                num_workers=1)
        stopping_iteration = 0
        current_loss = np.inf
        self.model = AUTOENCODER(features.shape[1],
                                 hidden_size=self.hidden_layer_size,
                                 nn_type=self.nn_type,
                                 dropout=self.dropout,
                                 device=self.device).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.learning_rate)
        self.num_params = sum(p.numel() for p in self.model.parameters())
        if self.verbose:
            logging.info("Number of parameters {}".format(self.num_params))
            logging.info("Starting training for {} epochs".format(
                self.num_epochs))
        if self.verbose:
            pbar = tqdm.tqdm(total=self.stopping_crit)
        self.model.train()
        for epoch in range(self.num_epochs):
            if stopping_iteration > self.stopping_crit:
                if self.verbose:
                    logging.info("Stopping reached!")
                break
            losses_per_batch = []
            for i, (features) in enumerate(dataloader):
                features = features.float().to(self.device)
                outputs = self.model(features)
                features = torch.squeeze(features)
                loss = self.loss(outputs, features)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                losses_per_batch.append(float(loss))
            mean_loss = np.mean(losses_per_batch)
            if mean_loss < current_loss:
                current_loss = mean_loss
            else:
                if self.verbose:
                    pbar.update(1)
                stopping_iteration += 1
            if self.verbose:
                logging.info("epoch {}, mean loss per batch {}".format(
                    epoch, mean_loss))

    def transform(self, features):
        test_dataset = E2EDatasetLoader(features, None)
        predictions = []
        with torch.no_grad():
            for features in test_dataset:
                self.model.eval()
                features = features.float().to(self.device)
                representation = self.model.get_rep(features)
                pred = representation.detach().cpu().numpy()[0]
                predictions.append(pred)
        return np.matrix(predictions)

    def fit_transform(self, features):
        self.fit(features)
        return self.transform(features)


if __name__ == "__main__":

    X = np.random.random((100, 1000))
    X = sparse.csr_matrix(X)
    ae = GenericAutoencoder(n_components=16,
                            batch_size=32,
                            verbose=False,
                            nn_type="large")
    ae.fit(X)
    representation = ae.transform(X)
    print(representation.shape)
