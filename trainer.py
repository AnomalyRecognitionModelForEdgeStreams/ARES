import os.path
import random

import numpy as np
import torch.nn as nn
import torch
from torch_geometric.nn.models import GraphSAGE
from torch_geometric.nn import GAE
import matplotlib.pyplot as plt


class Trainer(nn.Module):
    def __init__(self, params):
        super(Trainer, self).__init__()

        self.params = params
        self.in_channels = params.in_channels
        self.hidden_channels = params.hidden_channels
        self.out_channels = params.out_channels
        self.num_layers = params.num_layers
        self.epochs = params.epochs
        self.patience = params.patience
        self.model_name = params.model_name
        self.learning_rate = params.learning_rate
        self.device = params.device
        self.dataset = params.dataset

        random.seed(params.seed)
        np.random.seed(params.seed)
        torch.manual_seed(params.seed)
        torch.cuda.manual_seed_all(params.seed)
        os.environ['PYTHONHASHSEED'] = str(params.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.model = None

        self.model = GAE(
            GraphSAGE(
                in_channels=self.in_channels,
                hidden_channels=self.hidden_channels,
                num_layers=self.num_layers,
                out_channels=self.out_channels)).to(
            self.device)

        self.optimizer = torch.optim.Adam(
            list(self.model.parameters()), lr=self.learning_rate)

    def train(self, train_graph):
        self.model.train()

        total_loss = 0.0
        batch = train_graph

        x = batch.x.to(self.device)
        edge_index = batch.edge_index.to(self.device)

        self.optimizer.zero_grad()

        z = self.model.encode(x=x, edge_index=edge_index)

        loss = self.model.recon_loss(z, edge_index)

        loss.backward()
        self.optimizer.step()
        total_loss += loss.detach()

        return float(total_loss)

    @torch.no_grad()
    def validation(self, val_graph):
        self.model.eval()

        total_loss = 0.0
        batch = val_graph

        x = batch.x.to(self.device)
        edge_index = batch.edge_index[:2, :].to(self.device)

        z = self.model.encode(x=x, edge_index=edge_index)

        loss = self.model.recon_loss(z, edge_index)

        total_loss += loss

        return float(total_loss)

    def fit(self, train_graph, val_graph):

        best_loss = np.inf

        if not os.path.exists("./results_losses"):
            os.makedirs("./results_losses")

        path_losses = f'./results_losses/losses_{self.model_name}_{self.dataset}.txt'
        path_val_losses = f'./results_losses/val_losses_{self.model_name}_{self.dataset}.txt'

        open(path_losses, 'w').close()
        open(path_val_losses, 'w').close()

        losses, val_losses = [], []
        patience_temp = 0

        try:

            for epoch in range(self.epochs):

                loss = self.train(train_graph)
                losses.append(loss)

                val_loss = self.validation(val_graph)
                val_losses.append(val_loss)

                with open(path_losses, 'a') as filehandle:
                    filehandle.write(f'\n {epoch}, {loss} \n')

                with open(path_val_losses, 'a') as filehandle:
                    filehandle.write(f'\n {epoch}, {val_loss} \n')

                if best_loss > val_loss:
                    best_loss = val_loss
                    patience_temp = 0
                    torch.save(
                        self.model.state_dict(), "{}.pt".format(
                            self.params.model_save_path))
                else:
                    patience_temp += 1

                print(
                    f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Loss Validation: {val_loss:.4f}, patience: {patience_temp}')

                if patience_temp == self.patience:
                    break

        finally:
            if val_graph is not None:
                print('---' * 30)
                print("BEST", best_loss)

                plt.figure(figsize=(15, 8))
                plt.plot(losses, label='training')
                plt.plot(val_losses, label='validation')
                plt.legend()
                plt.savefig(self.params.plot_title, bbox_inches='tight')
                plt.show()
