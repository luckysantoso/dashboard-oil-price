# Oil Price Dashboard

An interactive **Streamlit** application for visualizing historical crude oil prices (Brent/WTI) and generating future price forecasts using an **LSTM** model directly loaded within the dashboard.

---

## 🎯 Project Goals

* Provide an intuitive visual overview of historical oil price data.
* Integrate technical indicators (moving averages, Bollinger Bands, seasonal decomposition) for enhanced trend analysis.
* Offer a price prediction feature (1–30 days ahead) embedded in the Streamlit dashboard.

---

## ✨ Key Features

* **Interactive charts** with Plotly for daily prices.
* **Technical indicators**: moving averages, Bollinger Bands, seasonal decomposition.
* **Latest metrics**: current price, daily change, and moving averages.
* **Price forecasting** via LSTM model loaded directly in Streamlit.

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

   * Download historical price CSV from Yahoo Finance (ticker BZ=F or CL=F) as of **May 12, 2025**, and save as `data/oil_prices.csv`.
   * If you haven’t trained the model yet, run:

     ```bash
     python train.py
     ```

     This will generate `models/best_oil_lstm.pt` and `models/scaler.save`.

---

## ⚙️ Running the Application

Simply launch the Streamlit dashboard:

```bash
streamlit run app/dashboard.py
```

Open your browser at `http://localhost:8501` to use the dashboard.

---

## 📁 Project Structure

```plaintext
dashboard-oil-price/
├─ app/
│  └─ dashboard.py     # Streamlit dashboard application (loads model & scaler)
├─ data/
│  └─ oil_prices.csv   # Historical price data (CSV)
├─ models/
│  ├─ best_oil_lstm.pt # Trained LSTM model weights
│  └─ scaler.save      # Fitted MinMaxScaler object
├─ train.py            # Script to train the LSTM model
├─ requirements.txt    # Python package dependencies
└─ README.md           # Project documentation
```
