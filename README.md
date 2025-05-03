# Stock Price Prediction and Feature Impact using Linear Regression

An interactive Streamlit web app for predicting stock prices and visualizing feature impact using Linear Regression. Upload your own dataset or fetch real-time stock data, step through a complete ML pipeline, and enjoy a modern UI with finance-themed animations.

## 🚀 Features
- Upload your own CSV dataset or fetch data by stock ticker (via yfinance)
- Step-by-step ML pipeline: data loading, preprocessing, feature engineering, train/test split, model training, evaluation, and prediction
- Feature importance visualization (bar chart of model coefficients)
- Prediction vs. actual and future price charts (Plotly)
- Robust error handling for any dataset
- Customizable target column for prediction
- Modern dark UI with finance-themed Lottie animations and celebratory finish

## 🛠️ Setup
1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd <your-repo-folder>
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the app:**
   ```bash
   python -m streamlit run stock_price_app.py
   ```
4. **Open in your browser:**
   Visit [http://localhost:8501](http://localhost:8501)

## 📊 Usage
- **Upload a CSV:** Use the sidebar to upload your stock dataset (ensure it has columns like `Date`, `Close`, `Open`, `High`, `Low`, `Volume`).
- **Or fetch by ticker:** Enter a stock ticker (e.g., `AAPL`) and fetch 1 year of data from yfinance.
- **Step through the ML pipeline:** Click each step's button to process data, engineer features, split, train, evaluate, and predict.
- **Visualize results:** See feature impact, model performance, and future predictions with interactive charts.
- **Celebrate:** Enjoy a congratulatory animation when you complete all steps!

## 📝 Example CSV Format
```
Date,Close,Open,High,Low,Volume
2024-04-01,100,105,99,104,1000000
...
```

## 🤝 Credits
- Built with [Streamlit](https://streamlit.io/), [scikit-learn](https://scikit-learn.org/), [yfinance](https://github.com/ranaroussi/yfinance), [Plotly](https://plotly.com/python/), and [LottieFiles](https://lottiefiles.com/).

## 📬 Contact
For questions, suggestions, or collaboration, feel free to open an issue or reach out on LinkedIn! 
