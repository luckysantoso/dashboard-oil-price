import streamlit as st
import pandas as pd
import plotly.graph_objs as go
from statsmodels.tsa.seasonal import seasonal_decompose

st.set_page_config(page_title="Oil Price Dashboard", layout="wide")
st.title("Oil Price Dashboard")

# --- Sidebar with Identity ---
st.sidebar.markdown("## Lucky Santoso")
st.sidebar.markdown("#### Machine Learning Engineer")
st.sidebar.markdown("---")
st.sidebar.markdown("**Contact:**\n- Email: lucky.sntso@gmail.com\n- GitHub: lucky-santoso")
st.sidebar.markdown("---")

@st.cache_data
def load_data(path):
    # Baca file, lewati 2 baris pertama metadata
    df_raw = pd.read_csv(
        path,
        skiprows=2,
        header=None,
        names=["Date", "Close"]
    )
    # Hilangkan row yang 'Date'-nya NaN atau tidak sesuai format tanggal
    df_raw = df_raw[pd.to_datetime(df_raw["Date"], errors="coerce").notna()]
    # Convert kolom Date ke datetime
    df_raw["Date"] = pd.to_datetime(df_raw["Date"])
    # Set sebagai index dan urutkan
    df = df_raw.set_index("Date").sort_index()
    return df

df = load_data("data/oil_prices.csv")
result = seasonal_decompose(df["Close"], model="multiplicative", period=252)

last_price = df["Close"].iloc[-1]
prev_price = df["Close"].iloc[-2]
delta = last_price - prev_price
pct = delta / prev_price * 100
ma7 = df["Close"].rolling(7).mean().iloc[-1]
ma30 = df["Close"].rolling(30).mean().iloc[-1]

col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Harga Terkini", f"${last_price:.2f}")
col_b.metric("Perubahan Harian", f"{delta:.2f}", f"{pct:.2f}%")
col_c.metric("MA 7-hari", f"${ma7:.2f}")
col_d.metric("MA 30-hari", f"${ma30:.2f}")


st.subheader("Grafik Harga Historis")
fig_hist = go.Figure()
fig_hist.add_trace(
    go.Scatter(x=df.index, y=df["Close"], name="Harga Close", mode="lines")
)
fig_hist.update_layout(
    xaxis_title="Tanggal",
    yaxis_title="Harga (USD)",
    margin=dict(l=40, r=40, t=40, b=40),
    height=400
)
st.plotly_chart(fig_hist, use_container_width=True)
st.markdown("---")

st.subheader("Form Prediksi")
col1, col2 = st.columns([1, 1])

# Dapatkan nilai last date sebagai datetime.date
last_timestamp = pd.to_datetime(df.index).max()

with col1:
    start_date = st.date_input(
        "Mulai Prediksi",
        value=last_timestamp.to_pydatetime().date(),
        help="Pilih tanggal terakhir data historis sebagai titik awal prediksi"
    )
with col2:
    horizon = st.slider(
        "Horizon Prediksi (hari)",
        min_value=1,
        max_value=30,
        value=7,
        help="Jumlah hari ke depan untuk diprediksi"
    )

# Tombol untuk menjalankan prediksi
if st.button("Jalankan Prediksi"):
    import requests
    url = "http://localhost:8000/predict"
    payload = {"start_date": start_date.isoformat(), "horizon": horizon}
    try:
        res = requests.post(url, json=payload)
        res.raise_for_status()
        data = res.json()
        # Buat DataFrame hasil prediksi
        df_pred = pd.DataFrame({
            "Date": pd.to_datetime(data["dates"]),
            "Predicted": data["predictions"]
        }).set_index("Date")
        # Gabungkan dan plot
        df_plot = pd.concat([df["Close"], df_pred["Predicted"]], axis=1)
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(x=df_plot.index, y=df_plot["Close"], name="Actual"))
        fig_pred.add_trace(go.Scatter(x=df_plot.index, y=df_plot["Predicted"], name="Predicted"))
        st.subheader("Hasil Prediksi Harga Minyak")
        st.plotly_chart(fig_pred, use_container_width=True)
    except Exception as e:
        st.error(f"Gagal memanggil API prediksi: {e}")
else:
    st.info("Tekan tombol 'Jalankan Prediksi' untuk menjalankan model dan melihat hasil.")

st.markdown("---")

st.subheader("Data Terbaru")
# Hitung rolling stats
df["MA20"] = df["Close"].rolling(20).mean()
df["STD20"] = df["Close"].rolling(20).std()
df["BB_upper"] = df["MA20"] + 2 * df["STD20"]
df["BB_lower"] = df["MA20"] - 2 * df["STD20"]

# Saat plotting:
fig_hist.add_trace(go.Scatter(x=df.index, y=df["MA20"], name="MA20", line=dict(dash="dash")))
fig_hist.add_trace(go.Scatter(x=df.index, y=df["BB_upper"], name="Upper Band", line=dict(color="lightgrey")))
fig_hist.add_trace(go.Scatter(x=df.index, y=df["BB_lower"], name="Lower Band", line=dict(color="lightgrey"), fill="tonexty"))

st.dataframe(df.tail(10))

st.markdown(
    """
    **Catatan:**  
    - Data diambil dari Yahoo Finance pada tanggal **12 Mei 2025**  
    - Powered by: *Lucky Santoso*
    """
)