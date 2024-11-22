import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import DataLoader

from dataset import UltrasoundDataset
from architectures import Autoencoder
from architectures import AutoencoderWithSkipConnections
from losses import DAELosses, WeightedLoss

if __name__ == "__main__":
    model = AutoencoderWithSkipConnections()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    dataset = UltrasoundDataset("dataset", transforms=transform)

    train_indices, val_indices = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    losses = WeightedLoss(alpha=20.0)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 200
    for epoch in range(num_epochs):
        model.train()
        for batch in train_loader:
            annotated = batch['annotated']
            clean = batch['clean']

            optimizer.zero_grad()

            predicted = model(annotated)

            loss = losses(clean, annotated, predicted)

            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch}, Train Loss: {loss.item()}")

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                annotated = batch['annotated']
                clean = batch['clean']

                predicted = model(annotated)

                loss = losses(clean, annotated, predicted)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f"Epoch {epoch}, Validation Loss: {val_loss}")

    torch.save(model.state_dict(), "model.pt")
