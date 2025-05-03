import streamlit as st
import yfinance as yf
from PIL import Image
import requests
from io import BytesIO
from streamlit_lottie import st_lottie
import plotly.graph_objs as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# Set page config
st.set_page_config(page_title="Stock Price Prediction and Feature Impact using Linear Regression", layout="centered")

# Custom dark finance-themed background color using CSS
st.markdown(
    """
    <style>
    body {
        background-color: #181c20;
        color: #ffffff;
    }
    .stApp {
        background-color: #181c20;
        color: #ffffff;
    }
    .st-bw {
        background-color: #232a2f !important;
    }
    .stButton>button {
        background-color: #21ce99;
        color: #fff;
        border-radius: 8px;
        border: none;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #16a673;
        color: #fff;
    }
    .stSidebar {
        background-color: #232a2f;
    }
    .css-1d391kg, .css-1v0mbdj, .css-1cpxqw2 {
        color: #21ce99 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title and welcome message
st.title("🎉 Stock Price Prediction and Feature Impact using Linear Regression")
st.markdown("Welcome to the Stock Price Prediction app! Upload your Kragle dataset or fetch stock data by ticker.")

# Lottie finance animation (robust, will not crash app)
lottie_url = "https://assets10.lottiefiles.com/packages/lf20_jcikwtux.json"
try:
    lottie_response = requests.get(lottie_url, timeout=5)
    if lottie_response.status_code == 200 and lottie_response.content and lottie_response.content.strip():
        try:
            lottie_json = lottie_response.json()
            st_lottie(lottie_json, height=300, key="finance_lottie")
        except Exception as e:
            st.warning(f"Error decoding animation JSON: {e}. App will continue.")
    else:
        st.warning("Could not load Lottie animation (empty or invalid response). App will continue.")
except Exception as e:
    st.warning(f"Error loading animation: {e}. App will continue.")

# App details and instructions
st.markdown(
    """
    ## 📊 About This App
    This application allows you to:
    - **Predict stock prices** using Linear Regression.
    - **Visualize feature impact** on stock price predictions.
    - **Upload your own Kragle dataset** (CSV) for custom analysis.
    - **Fetch real-time stock data** by entering a ticker (e.g., `AAPL`).

    ### 🚀 How to Use
    1. **Upload a Kragle dataset** via the sidebar, or
    2. **Enter a stock ticker** and click 'Fetch Data from yfinance'.
    3. Preview your data instantly.
    4. Further features (modeling, visualization) will appear as you proceed.

    > *Empowering your financial insights with data and AI!*
    """
)

# Sidebar for navigation and actions
st.sidebar.header("Options")

# --- Data Cleaning Function for Uploaded CSVs ---
def clean_uploaded_df(df):
    # Strip whitespace from all column names
    df.columns = df.columns.str.strip()
    # Rename columns for consistency
    df = df.rename(columns={
        'Vol.': 'Volume',
        'Change %': 'Change',
        'Close ': 'Close',
        'close': 'Close',
        'CLOSE': 'Close',
    })
    # Clean Volume: convert K/M to numbers
    def parse_volume(val):
        if isinstance(val, str):
            val = val.replace(',', '').strip()
            if 'K' in val:
                return float(val.replace('K', '')) * 1_000
            elif 'M' in val:
                return float(val.replace('M', '')) * 1_000_000
            else:
                try:
                    return float(val)
                except:
                    return None
        return val
    if 'Volume' in df.columns:
        df['Volume'] = df['Volume'].apply(parse_volume)
    # Clean Change: remove % and convert to float
    if 'Change' in df.columns:
        df['Change'] = df['Change'].astype(str).str.replace('%', '').astype(float)
    return df

# Upload Kragle dataset
uploaded_file = st.sidebar.file_uploader("📁 Upload Kragle dataset (.csv)", type=["csv"])

# Input for stock ticker
st.sidebar.markdown("---")
ticker = st.sidebar.text_input("🔍 Enter a stock ticker (e.g., 'AAPL'):")
fetch_data = st.sidebar.button("Fetch Data from yfinance")

if fetch_data and ticker:
    with st.spinner(f"Fetching data for {ticker}..."):
        try:
            stock_data = yf.download(ticker, period="1y")
            if not stock_data.empty:
                st.success(f"Data for {ticker} loaded!")
                st.dataframe(stock_data.head())
            else:
                st.error("No data found for this ticker.")
        except Exception as e:
            st.error(f"Error fetching data: {e}")

if uploaded_file is not None and 'uploaded_df' not in st.session_state:
    try:
        df = pd.read_csv(uploaded_file)
        df = clean_uploaded_df(df)
        st.write("Columns after cleaning:", df.columns.tolist())  # Debug print
        if 'Close' not in df.columns:
            st.error("The 'Close' column is missing after cleaning. Please check your CSV headers.")
        st.session_state.uploaded_df = df.copy()
        st.success("Kragle dataset uploaded!")
        st.dataframe(df.head())
    except pd.errors.EmptyDataError:
        st.error("The uploaded file is empty. Please upload a valid CSV file with data.")
    except Exception as e:
        st.error(f"Error reading the uploaded file: {e}")

# Footer
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background: #232a2f;
        color: #21ce99;
        text-align: center;
        padding: 10px 0 8px 0;
        font-size: 16px;
        z-index: 100;
    }
    </style>
    <div class="footer">
        © 2025 Stock Price Prediction App | 💹 Powered by Finance & AI
    </div>
    """,
    unsafe_allow_html=True
)

# --- ML Pipeline Section ---
st.markdown("---")
st.header("⚙️ Step-by-Step ML Pipeline")

# State management for steps
if 'step' not in st.session_state:
    st.session_state.step = 0
if 'data' not in st.session_state:
    st.session_state.data = None
if 'features' not in st.session_state:
    st.session_state.features = []
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'y_pred' not in st.session_state:
    st.session_state.y_pred = None

# Step 1: Load Data
if st.button("✅ Step 1: Load Data"):
    if uploaded_file is not None and 'uploaded_df' in st.session_state:
        df = st.session_state.uploaded_df.copy()
        st.session_state.data = df.copy()
        st.success("Data loaded from uploaded file!")
        st.dataframe(df.head())
        st.session_state.step = 1
    elif fetch_data and ticker:
        stock_data = yf.download(ticker, period="1y")
        if not stock_data.empty:
            st.session_state.data = stock_data.copy()
            st.success(f"Data loaded for {ticker}!")
            st.dataframe(stock_data.head())
            st.session_state.step = 1
        else:
            st.warning("No data found for this ticker.")
    else:
        st.warning("Please upload a dataset or fetch stock data.")

# Step 2: Preprocessing
if st.session_state.step >= 1 and st.button("🧹 Step 2: Preprocessing"):
    df = st.session_state.data.copy()
    before = len(df)
    df = df.drop_duplicates()
    df = df.dropna()
    after = len(df)
    st.session_state.data = df.copy()
    st.success(f"Preprocessing complete! Removed {before - after} rows with missing values or duplicates.")
    st.dataframe(df.head())
    st.session_state.step = 2

# Step 3: Feature Engineering
if st.session_state.step >= 2 and st.button("🧠 Step 3: Feature Engineering"):
    df = st.session_state.data.copy()
    if 'Close' in df.columns:
        df['7d_MA'] = df['Close'].rolling(window=7).mean()
        df['Daily_Return'] = df['Close'].pct_change()
    if 'Volume' not in df.columns and 'Volume' in df:
        df['Volume'] = df['Volume']
    df = df.dropna()
    st.session_state.data = df.copy()
    st.session_state.features = [col for col in df.columns if col not in ['Close', 'Date']]
    st.success("Feature engineering complete!")
    st.markdown("**Features:** " + ", ".join(st.session_state.features))
    st.dataframe(df.head())
    st.session_state.step = 3

# Step 4: Train/Test Split
if st.session_state.step >= 3 and st.button("🔀 Step 4: Train/Test Split"):
    df = st.session_state.data.copy()
    target_col = 'Close'
    if target_col not in df.columns:
        # If 'Close' is missing, let user select a numeric column as target
        numeric_cols = df.select_dtypes(include=[float, int]).columns.tolist()
        if numeric_cols:
            target_col = st.selectbox(
                "'Close' column not found. Please select the target column for prediction:",
                numeric_cols,
                key="target_col_selectbox"
            )
            st.info(f"Using '{target_col}' as the target column.")
        else:
            st.error("Your data does not contain a 'Close' column or any numeric columns suitable for prediction. Please upload a valid file.")
            st.stop()
    X = df[st.session_state.features]
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    st.session_state.X_train = X_train
    st.session_state.X_test = X_test
    st.session_state.y_train = y_train
    st.session_state.y_test = y_test
    st.session_state.target_col = target_col
    st.success("Train/Test split complete!")
    # Pie chart
    split_labels = ['Train', 'Test']
    split_values = [len(X_train), len(X_test)]
    fig = go.Figure(data=[go.Pie(labels=split_labels, values=split_values, hole=.3)])
    fig.update_layout(title="Train/Test Split")
    st.plotly_chart(fig)
    st.session_state.step = 4

# Step 5: Model Training
if st.session_state.step >= 4 and st.button("📈 Step 5: Model Training"):
    model = LinearRegression()
    model.fit(st.session_state.X_train, st.session_state.y_train)
    st.session_state.model = model
    st.success("Linear Regression model trained!")
    st.session_state.step = 5

# Step 6: Evaluation
if st.session_state.step >= 5 and st.button("🧪 Step 6: Evaluation"):
    y_pred = st.session_state.model.predict(st.session_state.X_test)
    st.session_state.y_pred = y_pred
    r2 = r2_score(st.session_state.y_test, y_pred)
    mse = mean_squared_error(st.session_state.y_test, y_pred)
    st.success(f"Evaluation complete! R² Score: {r2:.4f}, MSE: {mse:.4f}")
    # Prediction vs Actual chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=st.session_state.y_test.values, mode='lines', name='Actual'))
    fig.add_trace(go.Scatter(y=y_pred, mode='lines', name='Predicted'))
    fig.update_layout(title="Prediction vs Actual (Test Set)", xaxis_title="Index", yaxis_title=st.session_state.target_col)
    st.plotly_chart(fig)
    st.session_state.step = 6

# Step 7: Feature Impact
if st.session_state.step >= 6 and st.button("📊 Step 7: Feature Impact"):
    coefs = st.session_state.model.coef_
    features = st.session_state.X_train.columns
    fig = go.Figure([go.Bar(x=features, y=coefs)])
    fig.update_layout(title="Feature Impact (Model Coefficients)", xaxis_title="Feature", yaxis_title="Coefficient")
    st.plotly_chart(fig)
    st.success("Feature impact visualized!")
    st.session_state.step = 7

# Step 8: Prediction
if st.session_state.step >= 7 and st.button("🔮 Step 8: Prediction"):
    # Predict next 5 days if possible, else show test set prediction
    df = st.session_state.data.copy()
    last_X = df[st.session_state.features].iloc[-5:]
    preds = st.session_state.model.predict(last_X)
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=df[st.session_state.target_col].values, mode='lines', name='Historical'))
    pred_indices = list(range(len(df), len(df)+5))
    fig.add_trace(go.Scatter(x=pred_indices, y=preds, mode='lines+markers', name='Next 5 Days Prediction'))
    fig.update_layout(title="Next 5 Days Prediction", xaxis_title="Index", yaxis_title=st.session_state.target_col)
    st.plotly_chart(fig)
    st.success("Prediction for next 5 days complete!")

    # Show congratulatory animation and message
    congrats_lottie_url = "https://assets2.lottiefiles.com/packages/lf20_jbrw3hcz.json"  # Celebration animation
    try:
        congrats_response = requests.get(congrats_lottie_url, timeout=5)
        if congrats_response.status_code == 200 and congrats_response.content and congrats_response.content.strip():
            try:
                congrats_json = congrats_response.json()
                st_lottie(congrats_json, height=250, key="congrats_lottie")
            except Exception as e:
                st.info("🎉 Congratulations! All steps completed.")
        else:
            st.info("🎉 Congratulations! All steps completed.")
    except Exception as e:
        st.info("🎉 Congratulations! All steps completed.")
    st.markdown("<h3 style='color:#21ce99;text-align:center;'>🎉 Congratulations! You have completed the full ML pipeline! 🎉</h3>", unsafe_allow_html=True) 