# app/api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import torch
import joblib
from train import OilPriceLSTM  # pastikan class model bisa diimpor

app = FastAPI()

# Load model & scaler sekali saat startup
try:
    model = OilPriceLSTM()
    model.load_state_dict(torch.load("models/best_oil_lstm.pt", map_location="cpu"))
    model.eval()
    scaler = joblib.load("models/scaler.save")
except Exception as e:
    raise RuntimeError(f"Gagal load model/scaler: {e}")

# Pydantic schema
class PredictRequest(BaseModel):
    start_date: str   # "YYYY-MM-DD"
    horizon: int      # jumlah hari prediksi

@app.post("/predict")
def predict(req: PredictRequest):
    try:
        # 1) Muat data historis
        df = pd.read_csv(
            "data/oil_prices.csv",
            skiprows=2,
            header=None,
            names=["Date", "Close"]
        )
        df = df[pd.to_datetime(df["Date"], errors="coerce").notna()]
        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)

        # 2) Ambil window terakhir hingga start_date
        if req.start_date not in df.index:
            # jika user pilih hari non-trading, ambil data sebelum itu
            df_slice = df.loc[:req.start_date]
        else:
            df_slice = df.loc[:req.start_date]
        if len(df_slice) < 30:
            raise HTTPException(status_code=400, detail="Tidak cukup data historis untuk titik mulai prediksi.")
        window = df_slice["Close"].values[-30:]

        # 3) Scale & iterasi prediksi
        seq = scaler.transform(window.reshape(-1,1)).flatten().tolist()
        preds_norm = []
        for _ in range(req.horizon):
            x = torch.tensor(seq[-30:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
            with torch.no_grad():
                p = model(x).item()
            preds_norm.append(p)
            seq.append(p)

        # 4) Inverse scale
        preds = scaler.inverse_transform([[v] for v in preds_norm]).flatten().tolist()

        # 5) Hitung tanggal prediksi tanpa `closed`
        start_ts = pd.to_datetime(req.start_date)
        # prediksi untuk hari berikutnya s.d. horizon
        dates = pd.date_range(
            start=start_ts + pd.Timedelta(days=1),
            periods=req.horizon
        )

        return {
            "dates": [d.strftime("%Y-%m-%d") for d in dates],
            "predictions": preds
        }

    except HTTPException:
        # biarkan HTTPException diteruskan
        raise
    except Exception as e:
        # tangani error lain
        raise HTTPException(status_code=500, detail=str(e))
