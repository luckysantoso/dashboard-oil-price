import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm.auto import tqdm

# --- Parameter Setting ---
WINDOW_SIZE = 30  # days for input sequence
HORIZON = 1       # days to predict ahead
BATCH_SIZE = 32
LR = 1e-3
EPOCHS = 100
PATIENCE = 5
MODEL_DIR = "models"
DATA_PATH = "data/oil_prices.csv"

# --- Dataset and DataLoader ---
class OilDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# --- Model Definition ---
class OilPriceLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.fc(last)

# --- Data Preparation Function ---
def load_and_prepare_data(path):
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
    prices_norm = scaler.fit_transform(df[["Close"]].values)

    X, y = [], []
    for i in range(len(prices_norm) - WINDOW_SIZE - HORIZON + 1):
        X.append(prices_norm[i:i+WINDOW_SIZE])
        y.append(prices_norm[i+WINDOW_SIZE:i+WINDOW_SIZE+HORIZON])
    X = np.array(X)
    y = np.array(y).squeeze()
    dataset = OilDataset(X, y)
    n_val = int(len(dataset) * 0.2)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    return train_loader, val_loader, scaler

# --- Training Function ---
def train_model(train_loader, val_loader, scaler):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OilPriceLSTM().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.L1Loss()

    best_val_mae = float("inf")
    epochs_no_improve = 0

    for epoch in tqdm(range(EPOCHS), desc="Epochs"):
        model.train()
        train_losses = []
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            y_pred = model(x_batch)
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_preds, val_trues = [], []
        with torch.inference_mode():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                out = model(x_batch)
                val_preds.extend(out.cpu().numpy().flatten())
                val_trues.extend(y_batch.cpu().numpy().flatten())

        val_mae = mean_absolute_error(val_trues, val_preds)
        val_rmse = np.sqrt(mean_squared_error(val_trues, val_preds))
        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Train MAE: {np.mean(train_losses):.4f} | "
            f"Val MAE: {val_mae:.4f} | Val RMSE: {val_rmse:.4f}"
        )

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            epochs_no_improve = 0
            os.makedirs(MODEL_DIR, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "best_oil_lstm.pt"))
            joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.save"))
            print("Saved best model & scaler.")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"No improvement for {PATIENCE} epochs - Early stopping.")
                break
    return model, scaler

# --- Main Guard ---
if __name__ == "__main__":
    train_loader, val_loader, scaler = load_and_prepare_data(DATA_PATH)
    train_model(train_loader, val_loader, scaler)
    print("Training completed.")
