import streamlit as st
from datetime import date
import yfinance as yf
import pandas as pd

from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

st.title("Stock Analysis")


stock_ticker_name = pd.read_csv("nasdaq_screener_1705012010431.csv", usecols=["Name"])
selected_stock_name = st.selectbox(
    "Select/Enter Company for Stock Analysis", stock_ticker_name
)
df = pd.read_csv("nasdaq_screener_1705012010431.csv")
tempStockTicker = df[df["Name"] == selected_stock_name]
selected_stock = tempStockTicker["Symbol"].values[0]


TODAY = date.today().strftime("%Y-%m-%d")
START = 2000


@st.cache_data()
def load_data(ticker, START):
    data = yf.download(ticker, str(START) + "-01-01", TODAY)
    data.reset_index(inplace=True)
    return data


year = st.slider(
    "Start Year:", 2000, 2024, on_change=load_data, args=(selected_stock, START)
)
data = load_data(selected_stock, year)


def plot_raw_data(start_year):
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=data["Date"], y=data["Open"], name="Stock Open"))
    figure.add_trace(go.Scatter(x=data["Date"], y=data["Close"], name="Stock Close"))
    figure.layout.update(xaxis_rangeslider_visible=True)
    st.plotly_chart(figure)


st.header(selected_stock_name + " (" + selected_stock + ")")
st.subheader("Current")
plot_raw_data(year)
st.write(data.tail())


st.header(selected_stock_name + " (" + selected_stock + ")")
st.subheader("Forecast")
n_years = st.slider("Years of Prediction:", 1, 4)
period = n_years * 365

df_training = data[["Date", "Close"]]
df_training = df_training.rename(columns={"Date": "ds", "Close": "y"})
m = Prophet()
m.fit(df_training)
future = m.make_future_dataframe(periods=period)
forcast = m.predict(future)


fig1 = plot_plotly(m, forcast)
st.plotly_chart(fig1)
