import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from numpy import array
from keras.utils import set_random_seed
from sklearn.preprocessing import MinMaxScaler
import keras
from keras.models import Sequential
import time
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error

def Train_test_split(sequence, n_steps, train_size, test_size):
    data_train = sequence[:train_size]
    data_test = sequence[train_size:train_size+test_size]
    data_valid = sequence[train_size+test_size:]
    
    X_train, X_test, X_valid, y_train, y_test, y_valid = [], [], [], [], [], []
    for i in range(len(data_train) - n_steps):
        X_train.append(data_train[i:i+n_steps]) 
        y_train.append(data_train[i+n_steps])
        
    for i in range(len(data_test) - n_steps):
        X_test.append(data_test[i:i+n_steps]) 
        y_test.append(data_test[i+n_steps])
        
    for i in range(len(data_valid) - n_steps):
        X_valid.append(data_valid[i:i+n_steps]) 
        y_valid.append(data_valid[i+n_steps])
    
    return array(X_train), array(X_test), array(X_valid), array(y_train), array(y_test), array(y_valid)

class CustomCallback(keras.callbacks.Callback):
    def __init__(self):
        self.my_bar = st.progress(0.0, text=str('0%'))
        self.time = time.time()
    
    def on_train_begin(self, logs=None):
        keys = list(logs.keys())

    def on_train_end(self, logs=None):
        keys = list(logs.keys())

    def on_epoch_begin(self, epoch, logs=None):
        keys = list(logs.keys())

    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())
        self.my_bar.progress((epoch+1)*(1.0/10), text=f'Progress: {int((epoch+1)*(1.0/10)*100)}% \n\n Time: {int(time.time()-self.time)}s')
        
def LSTM_model(X_train, y_train, X_valid, y_valid):
    set_random_seed(42)
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(64, return_sequences = False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    # fit model
    #early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='min', restore_best_weights=True)
    model.fit(X_train, y_train, epochs=10, verbose=1, validation_data=(X_valid, y_valid), callbacks=[CustomCallback()])
    return model

def Eval_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_inverted = scaler.inverse_transform(y_pred.reshape(y_pred.shape[0],-1))
    y_test_inverted = scaler.inverse_transform(y_test.reshape(y_test.shape[0],-1))
    rmse = round(mean_squared_error(y_test_inverted, y_pred_inverted, squared=False),1)
    mape = round(mean_absolute_percentage_error(y_test_inverted, y_pred_inverted)*100,2)
    mae = round(mean_absolute_error(y_test_inverted, y_pred_inverted),1)
    return y_pred_inverted, rmse, mape, mae

st.set_page_config(
    page_title="Forecasting time series",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state='collapsed'
)

st.markdown(
        f"""
        <style>
        [data-testid='stAppViewContainer'] {{
            background-image: url("https://cdn.pixabay.com/photo/2017/07/01/19/48/background-2462431_1280.jpg");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        [data-testid='stHeader'] {{
            background-color: rgba(0,0,0,0);
        }}
        </style>
        """,
        unsafe_allow_html=True)



st.markdown("<h1 style='text-align: center; color: black;'>Forecasting time series using LSTM model</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-left: center; font-size:30px; color: black;'>Choose a file (Excel or CSV)</h1>", unsafe_allow_html=True)
uploaded_file = st.file_uploader('_', label_visibility='collapsed')
if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)
    except: df = pd.read_csv(uploaded_file)
    
    col1, col2 = st.columns([1,1])
    col1.markdown("<h1 style='text-left: center; font-size:30px; color: black;'>Select time series data</h1>", unsafe_allow_html=True)
    time_col = col1.radio('_',list(df.columns), label_visibility='collapsed')
    
    col2.markdown("<h1 style='text-left: center; font-size:30px; color: black;'>Select price data</h1>", unsafe_allow_html=True)
    price_col = col2.radio('_',list(df.columns)[1:], label_visibility='collapsed')
    
    df[time_col] = pd.to_datetime(df[time_col]) 
    data_price = df[[price_col]].values
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data_price)
    
    train_size = int(0.6*len(data_scaled))
    test_size = int(0.3*len(data_scaled))
    valid_size = len(data_scaled) - train_size - test_size
    time_step = 100
    X_train, X_test, X_valid, y_train, y_test, y_valid = Train_test_split(data_scaled, n_steps=time_step, train_size=train_size, test_size=test_size)
    
    if 'model' not in st.session_state:
        st.session_state.model = None
        
    if st.button('Train model ⚙️'):
        with st.spinner('Wait for it...'):
            st.session_state.model = LSTM_model(X_train, y_train, X_test, y_test)
            st.success('This is a success message!', icon="✅")
            
    if st.session_state.model:
        y_pred_test, rmse, mape, mae = Eval_model(st.session_state.model, X_test, y_test)
        y_pred_val, _, _, _ = Eval_model(st.session_state.model, X_valid, y_valid)
        col1, col2 = st.columns([1,1])
        
        col2.markdown("<h1 style='text-left: center; font-size:30px; color: black;'>Evaluation metrics</h1>", unsafe_allow_html=True)
        col2.write(pd.DataFrame([[rmse, mape, mae]], columns=['RMSE', 'MAPE(%)', 'MAE']))
        col1.markdown("<h1 style='text-left: center; font-size:30px; color: black;'>Select predict next day</h1>", unsafe_allow_html=True)
        days = col1.radio('_', ['5days', '10days', '15days', '20days', '25days', '30days'], horizontal=True, label_visibility='collapsed')
        index = ['5days', '10days', '15days', '20days', '25days', '30days'].index(days)
        days_next = [5,10,15,20,25,30][index]
        name_days = ['Predict_5days', 'Predict_10days', 'Predict_15days', 'Predict_20days', 'Predict_25days', 'Predict_30days']
        
        y_pred_next_days = []
        time_data = df['Date'][-1:]
        time_30days = pd.Series(pd.date_range(df['Date'][-1:].values[0], periods=days_next, freq='D'))
        temp_input = data_scaled[-time_step:]
        
        show_frame_price = []
        for i in range(days_next):
            pred_next_day = st.session_state.model.predict(temp_input[i:i+time_step].reshape(1,time_step,1), verbose=0)
            y_pred_next_days.append(int(scaler.inverse_transform(pred_next_day)[0][0]))
            temp_input = np.append(temp_input, pred_next_day)
            if (i+1) % 5 == 0:
                show_frame_price.append(y_pred_next_days[-5:])
                
        show_frame_days = ['Days 1-5', 'Days 5-10', 'Days 10-15', 'Days 15-20', 'Days 20-25', 'Days 25-30']
        st.write(pd.DataFrame(show_frame_price, index=show_frame_days[:days_next//5]))
        
        fig = go.Figure()
        fig.update_layout(width=1000, height=700, paper_bgcolor="#fff", plot_bgcolor="#fff")
        fig.add_trace(go.Scatter(x=df[time_col][:train_size], y=data_price[:train_size].flatten(), name='Train'))
        fig.add_trace(go.Scatter(x=df[time_col][train_size:train_size+test_size], y=data_price[train_size:train_size+test_size].flatten(), name='Test'))
        fig.add_trace(go.Scatter(x=df[time_col][train_size+100:train_size+test_size], y=y_pred_test.flatten(), name='Predict_test'))
        fig.add_trace(go.Scatter(x=df[time_col][train_size+test_size:], y=data_price[train_size+test_size:].flatten(), name='Valid'))
        fig.add_trace(go.Scatter(x=df[time_col][train_size+test_size+100:], y=y_pred_val.flatten(), name='Pred_valid'))
        fig.add_trace(go.Scatter(x=time_30days, y=y_pred_next_days, name=name_days[index]))
        st.plotly_chart(fig)
        
