# Oil Price Dashboard

An interactive **Streamlit** application for visualizing historical crude oil prices (Brent/WTI) and generating future price forecasts using an **LSTM** model served via **FastAPI**.

---

## 🎯 Project Goals

- Provide an intuitive visual overview of historical oil price data.
- Integrate technical indicators (moving averages, Bollinger Bands, seasonal decomposition) for enhanced trend analysis.
- Offer a daily price prediction feature (1–30 days ahead) to support decision-making.

---

## ✨ Key Features

- **Interactive charts** with Plotly for daily prices.
- **Technical indicators**: moving averages, Bollinger Bands, seasonal decomposition.
- **Latest metrics**: current price, daily change, and moving averages.
- **Price forecasting** via LSTM model with user-defined horizon.

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

   - **Source data**: Download historical price CSV from Yahoo Finance (ticker BZ=F for Brent or CL=F for WTI) as of **May 12, 2025**, and save as `data/oil_prices.csv`.csv\`.
   - If you haven’t trained the model yet, run:

     ```bash
     python train.py
     ```

   - This will generate `models/best_oil_lstm.pt` and `models/scaler.save`.

---

## ⚙️ Running the Application

1. **Start the FastAPI backend**

   ```bash
   python -m uvicorn app.api:app --reload
   ```

   - Access Swagger UI at `http://127.0.0.1:8000/docs`.

2. **Launch the Streamlit dashboard**

   ```bash
   streamlit run app/dashboard.py
   ```

   - Open your browser at `http://localhost:8501`.

---

## 📁 Project Structure

```plaintext
dashboard-oil-price/
├─ app/
│  ├─ api.py           # FastAPI endpoint for price predictions
│  └─ dashboard.py     # Streamlit dashboard application
├─ data/
│  └─ oil_prices.csv   # Historical price data (CSV)
├─ models/
│  ├─ best_oil_lstm.pt # Trained LSTM model weights
│  └─ scaler.save      # Fitted MinMaxScaler object
├─ train.py            # Script to train the LSTM model
├─ requirements.txt    # Python package dependencies
└─ README.md           # Project documentation
```

---

## 🤝 Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request with improvements or feature suggestions.

---

## 📬 Contact

**Lucky Santoso**
✉️ [lucky@example.com](mailto:lucky@example.com)
🔗 [github.com/luckysantoso](https://github.com/luckysantoso)

---

> **Note:** This project is for demonstration purposes. For production use, ensure thorough data validation and model retraining according to your business needs.
