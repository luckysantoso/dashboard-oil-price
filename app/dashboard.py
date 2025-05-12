import streamlit as st
import pandas as pd
import plotly.graph_objs as go
from statsmodels.tsa.seasonal import seasonal_decompose
import torch
import joblib
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from train import OilPriceLSTM  # pastikan path import ini sesuai struktur Anda
import yfinance as yf
import datetime as dt
from data.fetch_data import fetch_and_save

# —————————————— Streamlit Config ——————————————
st.set_page_config(page_title="Oil Price Dashboard", layout="wide")
st.title("Oil Price Dashboard")

# —————————————— Sidebar ——————————————
st.sidebar.markdown("## Lucky Santoso")
st.sidebar.markdown("#### Machine Learning Engineer")
st.sidebar.markdown("---")
st.sidebar.markdown("**Contact:**\n- Email: [lucky.sntso@gmail.com](mailto:lucky.sntso@gmail.com)\n- GitHub: lucky-santoso")
st.sidebar.markdown("---")

# —————————————— Data Loading ——————————————
@st.cache_data(ttl=24 * 60 * 60)
def load_data(ticker="BZ=F", period="10y", csv_path="data/oil_prices.csv"):
    # pastikan folder data ada & selalu fetch terbaru
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    fetch_and_save(tickers=ticker, period=period)

    df_raw = pd.read_csv(
        csv_path,
        skiprows=2,
        header=None,
        names=["Date", "Close"]
    )
    df_raw = df_raw[pd.to_datetime(df_raw["Date"], errors="coerce").notna()]
    df_raw["Date"] = pd.to_datetime(df_raw["Date"])
    return df_raw.set_index("Date").sort_index()


# Panggil
df = load_data("BZ=F", "10y")


# —————————————— Model & Scaler Loading ——————————————
@st.cache_resource
def load_model_and_scaler():
    # load LSTM model
    model = OilPriceLSTM()
    model.load_state_dict(torch.load("models/best_oil_lstm.pt", map_location="cpu"))
    model.eval()
    # load scaler
    scaler = joblib.load("models/scaler.save")
    return model, scaler

model, scaler = load_model_and_scaler()

# —————————————— Statistik Terkini ——————————————
last_price = float(df["Close"].iloc[-1])
prev_price = float(df["Close"].iloc[-2])
delta = last_price - prev_price
pct = delta / prev_price * 100
ma7 = float(df["Close"].rolling(7).mean().iloc[-1])
ma30 = float(df["Close"].rolling(30).mean().iloc[-1])

col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Harga Terkini", f"${last_price:.2f}")
col_b.metric("Perubahan Harian", f"{delta:.2f}", f"{pct:.2f}%")
col_c.metric("MA 7-hari", f"${ma7:.2f}")
col_d.metric("MA 30-hari", f"${ma30:.2f}")

# —————————————— Decompose & Plot Harga Historis ——————————————
result = seasonal_decompose(df["Close"], model="multiplicative", period=252)

st.subheader("Grafik Harga Historis")
fig_hist = go.Figure()
fig_hist.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Harga Close", mode="lines"))
fig_hist.update_layout(
    xaxis_title="Tanggal",
    yaxis_title="Harga (USD)",
    margin=dict(l=40, r=40, t=40, b=40),
    height=400
)
st.plotly_chart(fig_hist, use_container_width=True)
st.markdown("---")

# —————————————— Form Prediksi ——————————————
st.subheader("Form Prediksi")
last_timestamp = df.index.max()

col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input(
        "Mulai Prediksi",
        value=last_timestamp.date(),
        help="Tanggal terakhir data historis"
    )
with col2:
    horizon = st.slider(
        "Horizon Prediksi (hari)",
        min_value=1,
        max_value=30,
        value=7,
        help="Jumlah hari ke depan untuk diprediksi"
    )

if st.button("Jalankan Prediksi"):
    try:
        # ambil window 30 hari terakhir hingga start_date
        df_slice = df.loc[:pd.to_datetime(start_date)]
        if len(df_slice) < 30:
            st.error("Tidak cukup data historis (minimal 30 hari).")
        else:
            window = df_slice["Close"].values[-30:]
            seq = scaler.transform(window.reshape(-1, 1)).flatten().tolist()

            preds_norm = []
            for _ in range(horizon):
                x = torch.tensor(seq[-30:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
                with torch.no_grad():
                    p = model(x).item()
                preds_norm.append(p)
                seq.append(p)

            preds = scaler.inverse_transform([[v] for v in preds_norm]).flatten().tolist()
            dates = pd.date_range(
                start=pd.to_datetime(start_date) + pd.Timedelta(days=1),
                periods=horizon
            )

            df_pred = pd.DataFrame({"Date": dates, "Predicted": preds}).set_index("Date")
            df_plot = pd.concat([df["Close"], df_pred["Predicted"]], axis=1)

            # plot hasil prediksi
            fig_pred = go.Figure()
            fig_pred.add_trace(go.Scatter(x=df_plot.index, y=df_plot["Close"], name="Actual"))
            fig_pred.add_trace(go.Scatter(x=df_plot.index, y=df_plot["Predicted"], name="Predicted"))
            st.subheader("Hasil Prediksi Harga Minyak")
            st.plotly_chart(fig_pred, use_container_width=True)
    except Exception as e:
        st.error(f"Error saat prediksi: {e}")

st.markdown("---")

# —————————————— Data Terbaru & Bollinger Bands ——————————————
df["MA20"] = df["Close"].rolling(20).mean()
df["STD20"] = df["Close"].rolling(20).std()
df["BB_upper"] = df["MA20"] + 2 * df["STD20"]
df["BB_lower"] = df["MA20"] - 2 * df["STD20"]

fig_hist.add_trace(go.Scatter(x=df.index, y=df["MA20"], name="MA20", line=dict(dash="dash")))
fig_hist.add_trace(go.Scatter(x=df.index, y=df["BB_upper"], name="Upper Band", line=dict(dash="dash")))
fig_hist.add_trace(go.Scatter(x=df.index, y=df["BB_lower"], name="Lower Band", fill="tonexty", line=dict(dash="dash")))

st.dataframe(df.tail(10))

current_year = dt.date.today().year
st.markdown(
    f"""
    <div style="text-align: center; font-size:20px; color: gray; margin-top: 1rem;">
        Powered by Lucky Santoso &bull; {current_year}
    </div>
    """,
    unsafe_allow_html=True
)
