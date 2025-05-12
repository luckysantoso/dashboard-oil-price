# Oil Price Dashboard

An interactive **Streamlit** application for visualizing historical crude oil prices (Brent/WTI) and generating future price forecasts using a **N-BEATS** model directly loaded within the dashboard.

---

## 🎯 Project Goals

- Provide an intuitive visual overview of historical oil price data.
- Integrate technical indicators (moving averages, Bollinger Bands, seasonal decomposition) for enhanced trend analysis.
- Offer a price prediction feature (1–100 days ahead) embedded in the Streamlit dashboard, powered by **N-BEATS**.

---

## ✨ Key Features

- **Interactive charts** with Plotly for daily prices.
- **Technical indicators**: moving averages, Bollinger Bands, seasonal decomposition.
- **Latest metrics**: current price, daily change, and moving averages.
- **Price forecasting** via **N-BEATS** model loaded directly in Streamlit.

---

## 🚀 Installation & Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/luckysantoso/dashboard-oil-price.git
   cd dashboard-oil-price
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv venv
   # Windows
   .\venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare data & model**

   - Download historical price CSV from Yahoo Finance (ticker BZ=F or CL=F) as of **May 12, 2025**, and save as `data/oil_prices.csv`.
   - If you haven’t trained the model yet, run:

     ```bash
     python train.py
     ```

     This will generate `models/best_nbeats.pt` and `models/scaler.save`.

---

## ⚙️ Running the Application

Launch the Streamlit dashboard:

```bash
streamlit run app/dashboard.py
```

Open your browser at `http://localhost:8501` to use the dashboard.

---

## 📁 Project Structure

```plaintext
dashboard-oil-price/
├─ app/
│  └─ dashboard.py         # Streamlit
├─ data/
│  └─ oil_prices.csv       # Historical price
├─ models/
│  ├─ best_nbeats.pt       # Trained N-BEATS
│  └─ scaler.save          # Fitted
├─ train.py         # Script to train the
├─ requirements.txt        # Python package
└─ README.md               # Project
```

---

## 📝 Notes

- The N-BEATS architecture decomposes the input window into trend and/or seasonality components via stacked residual blocks and basis functions.
- You can adjust the forecast horizon (up to 100 days) and window size in the Streamlit UI.
- For further customization—basis functions, block depth, learning rate—edit `train.py` accordingly.

---

## 📚 Daftar Pustaka

1. Oreshkin, B. N., Carpov, D., Chapados, N., & Bengio, Y. (2019). _N-BEATS: Neural basis expansion analysis for interpretable time series forecasting_. arXiv preprint arXiv:1905.10437. https://doi.org/10.48550/arXiv.1905.10437
