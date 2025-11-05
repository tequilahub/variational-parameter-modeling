import warnings
warnings.filterwarnings('ignore')

import os

import torch
import torch.nn as nn
import torch.optim as optim

from .models.mixed_schnet import MixedSchNet
from .load_dataset import get_dataset, get_dataloader
from .evaluation import evaluate_structure

NUM_EPOCHS = 3
BATCH_SIZE = 1

def train():
    mse = nn.MSELoss()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    scaler = torch.amp.GradScaler(device)

    # Combine Datasets to complex one
    train_1 = get_dataset(fp=f"h4/a.csv")
    train_2 = get_dataset(fp=f"h4/b.csv")

    train_loader = get_dataloader([train_1, train_2], batch_size=BATCH_SIZE)

    # Define what the model shall be evaluated on
    eval_structure = {
        #"h4 random": get_dataloader([get_dataset(fp="h4/test_random_100.csv")], batch_size=BATCH_SIZE),
        #"h6 linear": get_dataloader([get_dataset(fp="h6/test_linear_100.csv")], batch_size=BATCH_SIZE),
        "h6 random": get_dataloader([get_dataset(fp="h6/test_random_100.csv")], batch_size=BATCH_SIZE),
        #"h8 linear": get_dataloader([get_dataset(fp="h8/test_linear_100.csv")], batch_size=BATCH_SIZE),
    }
    model = MixedSchNet().to(device)
    optimizer = optim.Adam(model.parameters())
    model.train()

    best_error = float("inf")
    for epoch in range(NUM_EPOCHS):
        for edges, geometry, target, _, _, _ in train_loader:
            optimizer.zero_grad()

            geometry = geometry.float().to(device)
            target = target.float().to(device)
            n, _ = geometry[0].shape

            z = torch.ones(n, dtype=torch.long, device=device).to(device)
            batch = torch.zeros(n, dtype=torch.long, device=device).to(device)

            params = model(z=z, pos=geometry[0], batch=batch, edges=edges)

            loss = mse(params, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        e = evaluate_structure(model, eval_structure, device, epoch)
        if e < best_error:
            torch.save(model.state_dict(), os.path.join(os.getcwd(), f"MixedSchnet.pth"))
            best_error = e
            print("Model saved")
        print("--------------------")
    return best_error

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    torch.cuda.empty_cache()

    train()
