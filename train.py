import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import joblib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm.auto import tqdm

# --- Parameter Setting ---
WINDOW_SIZE = 100
HORIZON     = 1
BATCH_SIZE  = 32
LR          = 1e-3
EPOCHS      = 100
PATIENCE    = 10
MODEL_DIR   = "models"
DATA_PATH   = "data/oil_prices.csv"
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class OilDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class GenericBasis(nn.Module):
    def __init__(self, backcast_size, forecast_size):
        super().__init__()
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size
    def forward(self, theta):
        return theta[:, :self.backcast_size], theta[:, -self.forecast_size:]


class NBeatsBlock(nn.Module):
    def __init__(self, input_size, theta_size, basis, n_layers, layer_size):
        super().__init__()
        self.mlp = nn.ModuleList(
            [nn.Linear(input_size, layer_size)] +
            [nn.Linear(layer_size, layer_size) for _ in range(n_layers-1)]
        )
        self.theta = nn.Linear(layer_size, theta_size)
        self.basis = basis
    def forward(self, x):
        h = x
        for lin in self.mlp:
            h = nn.functional.relu(lin(h))
        t = self.theta(h)
        return self.basis(t)


class NBeats(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
    def forward(self, x):
        x = x.squeeze(-1)
        residuals = x.flip(dims=(1,))
        forecast = torch.zeros(x.size(0), HORIZON, device=x.device)
        mask = torch.ones_like(residuals)
        for blk in self.blocks:
            back, f = blk(residuals)
            residuals = (residuals - back) * mask
            forecast = forecast + f
        return forecast


def load_data(path):
    df = pd.read_csv(
        path,
        skiprows=2,
        header=None,
        names=["Date", "Close"]
    )
    df = df[pd.to_datetime(df["Date"], errors="coerce").notna()]
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)

    scaler = MinMaxScaler()
    vals = scaler.fit_transform(df[["Close"]].values)

    X, y = [], []
    for i in range(len(vals) - WINDOW_SIZE - HORIZON + 1):
        X.append(vals[i : i + WINDOW_SIZE])
        y.append(vals[i + WINDOW_SIZE : i + WINDOW_SIZE + HORIZON])
    X = np.array(X)
    y = np.array(y).squeeze()

    dataset = OilDataset(X, y)
    n_val = int(len(dataset) * 0.2)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE)
    return train_loader, val_loader, scaler


def main():
    # Prepare data
    train_loader, val_loader, scaler = load_data(DATA_PATH)

    # Build N-Beats model
    backcast_size = WINDOW_SIZE
    theta_size    = WINDOW_SIZE + HORIZON
    basis_fn      = GenericBasis(backcast_size, HORIZON)
    blocks = [
        NBeatsBlock(
            input_size = backcast_size,
            theta_size = theta_size,
            basis      = basis_fn,
            n_layers   = 4,
            layer_size = 256
        )
        for _ in range(3)
    ]
    model = NBeats(blocks).to(DEVICE)

    # Loss & optimizer
    criterion_mse = nn.MSELoss()
    criterion_mae = nn.L1Loss()
    optimizer     = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_mae = float("inf")
    no_improve   = 0

    # Training loop
    for epoch in tqdm(range(1, EPOCHS+1), desc="Epochs"):
        model.train()
        running_mse, running_mae = 0.0, 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            y_pred = model(xb)
            loss_mse = criterion_mse(y_pred, yb)
            loss_mae = criterion_mae(y_pred, yb)
            loss_mse.backward()
            optimizer.step()
            running_mse += loss_mse.item() * xb.size(0)
            running_mae += loss_mae.item() * xb.size(0)
        train_mse = running_mse / len(train_loader.dataset)
        train_mae = running_mae / len(train_loader.dataset)

        model.eval()
        val_mse_sum, val_mae_sum = 0.0, 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                y_pred = model(xb)
                val_mse_sum += criterion_mse(y_pred, yb).item() * xb.size(0)
                val_mae_sum += criterion_mae(y_pred, yb).item() * xb.size(0)
        val_mse = val_mse_sum / len(val_loader.dataset)
        val_mae = val_mae_sum / len(val_loader.dataset)

        print(
            f"Epoch {epoch:02d} | "
            f"Train MSE: {train_mse:.6f}, MAE: {train_mae:.6f} | "
            f"Val   MSE: {val_mse:.6f}, MAE: {val_mae:.6f}"
        )

        # Early stopping
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            no_improve = 0
            os.makedirs(MODEL_DIR, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "best_nbeats.pt"))
            joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.save"))
            print("  → Saved best model & scaler.")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"No improvement in {PATIENCE} epochs. Stopping early.")
                break

    print("✅ Training completed.")


if __name__ == "__main__":
    main()
