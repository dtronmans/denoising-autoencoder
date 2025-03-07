from torch import nn
from tqdm import tqdm
import os

import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.config import Config
from src.losses import WeightedLoss

from torch.utils.tensorboard import SummaryWriter

if __name__ == "__main__":
    config = Config(os.path.join("src", "config.json"))
    writer = SummaryWriter()

    model = config.architecture
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_indices, val_indices = train_test_split(range(len(config.dataset)), test_size=config.val_split, random_state=42)
    train_dataset = torch.utils.data.Subset(config.dataset, train_indices)
    val_dataset = torch.utils.data.Subset(config.dataset, val_indices)

    print("Train dataset length: " + str(len(train_dataset)))
    print("Val dataset length: " + str(len(val_dataset)))

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    losses = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    num_epochs = config.epochs
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for batch in tqdm(train_loader):
            annotated = batch['annotated'].to(device)
            clean = batch['clean'].to(device)

            optimizer.zero_grad()
            predicted = model(annotated)
            loss = losses(clean, predicted)
            train_loss += loss.item()

            loss.backward()
            optimizer.step()

        writer.add_scalar("Denoising Loss/train", loss, epoch)
        loss = loss * 100
        print(f"Epoch {epoch}, Train Loss: {loss.item():.2f}")

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader):
                annotated = batch['annotated'].to(device)
                clean = batch['clean'].to(device)

                predicted = model(annotated)

                loss = losses(clean, predicted)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        writer.add_scalar("Denoising Loss/val", val_loss, epoch)
        val_loss = val_loss * 100
        print(f"Epoch {epoch}, Validation Loss: {val_loss:.2f}")

    torch.save(model.state_dict(), "model384_384.pt")
    writer.flush()
