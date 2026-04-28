import pickle

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from pick_a_pka.molgpka.utils.net import GCNNet


BATCH_SIZE = 128
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL= GCNNet().to(DEVICE)
OPTIMIZER = torch.optim.Adam(MODEL.parameters(), lr=0.0001)
SCHEDULER = torch.optim.lr_scheduler.ReduceLROnPlateau(OPTIMIZER, mode='min',
                                                       factor=0.7, patience=10,
                                                       min_lr=0.00001)


def load_data(file_name):
    with open(file_name, "rb") as f:
        conts = f.read()
    data = pickle.loads(conts)
    return data


def prepare_dataset():
    train_data = load_data("train.pickle")
    valid_data = load_data("valid.pickle")
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE)
    valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE)
    return train_loader, valid_loader


def train(epoch):
    MODEL.train()
    loss_all = 0

    for data in train_loader:
        data = data.to(DEVICE)
        OPTIMIZER.zero_grad()
        output = MODEL(data)
        loss = F.mse_loss(output, data.pka)
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        OPTIMIZER.step()
    return loss_all / len(train_loader.dataset)


def test(loader):
    MODEL.eval()
    correct = 0
    mae = 0
    for data in loader:
        data = data.to(DEVICE)
        output = MODEL(data)

        correct += F.mse_loss(output, data.pka).item() * data.num_graphs
        mae += F.l1_loss(output, data.pka).item() * data.num_graphs
    return correct / len(loader.dataset), mae / len(loader.dataset)


train_loader, valid_loader = prepare_dataset()


hist = {"loss":[], "mse":[], "mae":[]}
for epoch in range(1, 1001):
    PATH = "models/weight_{}.pth".format(epoch)
    train_loss = train(epoch)
    mse, mae = test(valid_loader)

    hist["loss"].append(train_loss)
    hist["mse"].append(mse)
    hist["mae"].append(mae)
    if mse <=  min(hist["mse"]):
        torch.save(MODEL.state_dict(), PATH)
    print(f'Epoch: {epoch}, Train loss: {train_loss:.3}, Test_mse: {mse:.3}, Test_mae: {mae:.3}')
