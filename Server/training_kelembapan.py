#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from keras.models import load_model, Sequential
from keras.layers import LSTM, Dense, Dropout
from sqlalchemy import create_engine
import os
import psycopg2

# Fungsi untuk menghubungkan ke database PostgreSQL dan mengambil data
def fetch_data_from_db():
    # Sesuaikan dengan kredensial database Anda
    user = 'postgres'
    password = '1'
    host = 'localhost'
    port = '5432'
    database = 'postgres'

    # Membuat koneksi ke database
    engine = create_engine(f'postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}')
    query = 'SELECT hum FROM datadht_client1'
    
    # Membaca data dari database ke dalam DataFrame
    data = pd.read_sql(query, engine)
    data = data[-8640:]
    # data = data['temp'] <= 27.5]
    # data = data[data['hum'] <= 27.3]
    return data

# Fungsi untuk membuat jendela geser
def create_sliding_window(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)

# Fungsi untuk membuat model LSTM
def LSTM_Model(trial):
    model = Sequential()
    
    # Parameter yang dioptimalkan
    n_layers = trial.suggest_int('n_layers', 1, 3)
    input_shape = (X_train.shape[1], X_train.shape[2])
    
    for i in range(n_layers):
        num_units = trial.suggest_int(f'num_units_l{i}', 50, 200)
        dropout_rate = trial.suggest_float(f'dropout_rate_l{i}', 0.1, 0.5)
        
        return_sequences = i < n_layers - 1
        model.add(LSTM(num_units, activation='relu', return_sequences=return_sequences, input_shape=input_shape if i == 0 else (None,)))
        model.add(Dropout(dropout_rate))
    
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(trial.suggest_float('dropout_rate_dense', 0.1, 0.5)))
    model.add(Dense(1))
    
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    return model


# In[2]:


# Panggil fungsi untuk mengambil data
data = fetch_data_from_db()
dht = data[['hum']].values


# In[3]:


dht


# In[4]:


pd.DataFrame(dht).plot()


# In[5]:


dht


# In[6]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import optuna
from optuna.integration import TFKerasPruningCallback
from tensorflow.keras.callbacks import EarlyStopping


# In[7]:


window_size = 10  # Ganti dengan nilai tetap atau tentukan sesuai kebutuhan
X, y = create_sliding_window(dht, window_size)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
input_shape = (X_train.shape[1], X_train.shape[2])


# In[8]:


X.shape


# In[9]:


# def objective(trial):
#     model = LSTM_Model(trial)
    
#     early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min')
#     model.fit(X_train, y_train, epochs=50, batch_size=trial.suggest_int('batch_size', 16, 64), 
#               validation_data=(X_test, y_test), callbacks=[TFKerasPruningCallback(trial, 'val_loss'), early_stopping], verbose=0)
    
#     val_loss = model.evaluate(X_test, y_test, verbose=0)
#     return val_loss


# In[10]:


# study = optuna.create_study(direction='minimize')
# study.optimize(objective, n_trials=50)

# print('Best trial:')
# trial = study.best_trial
# print(f'  Value: {trial.value}')
# print('  Params:')
# for key, value in trial.params.items():
#     print(f'    {key}: {value}')


# In[11]:


def create_best_LSTM_Model(input_shape, best_params):
    model = Sequential()
    
    for i in range(best_params['n_layers']):
        num_units = best_params[f'num_units_l{i}']
        dropout_rate = best_params[f'dropout_rate_l{i}']
        
        return_sequences = i < best_params['n_layers'] - 1
        model.add(LSTM(num_units, activation='relu', return_sequences=return_sequences, input_shape=input_shape if i == 0 else (None,)))
        model.add(Dropout(dropout_rate))
    
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(best_params['dropout_rate_dense']))
    model.add(Dense(1))
    
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    return model


# In[12]:


best_params = {
    'n_layers': 1,
    'num_units_l0': 91,
    'dropout_rate_l0': 0.21785263666275018,
    'dropout_rate_dense': 0.2803366278088336,
    'batch_size': 30
}


# In[13]:


model_path = 'model_hum.h5'


# In[ ]:





# In[14]:


# X, y = create_sliding_window(dht, window_size)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# input_shape = (X_train.shape[1], X_train.shape[2])
model = create_best_LSTM_Model(input_shape, best_params)

if os.path.exists(model_path):
    print("Loading existing model...")
    model = load_model(model_path)
else:
    print("Building new model...")
    model = create_best_LSTM_Model(input_shape, best_params)

model.fit(X_train, y_train, epochs=100, verbose=1)

predictions = model.predict(X_test)

mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse:.4f}')

r2 = r2_score(y_test, predictions)
print(f'R-squared Score: {r2:.4f}')


# In[15]:


def send_prediction_to_db(r2):
    conn = psycopg2.connect(
        dbname="postgres",
        user="postgres",
        password="1",
        host="localhost",
        port = "5432"
    )
    cursor = conn.cursor()
    
    # Insert the prediction data into a different table
    query = 'INSERT INTO "Hasil_trainning" (kelembapan) VALUES (%s)'
    cursor.execute(query, (r2,))
    
    conn.commit()
    cursor.close()
    conn.close()


# In[16]:


send_prediction_to_db(r2)


# In[17]:


# Menyimpan model yang sudah dilatih
model.save(model_path)
print(f'Model saved to {model_path}')


# In[18]:


import matplotlib.pyplot as plt
import pandas as pd

# Membuat DataFrame dari data
df_y_test = pd.DataFrame(y_test, columns=['y_test'])
df_predictions = pd.DataFrame(predictions, columns=['predictions'])


plt.figure(figsize=(20, 4))

# Plot y_test
plt.plot(y_test, label='y_test')
# Plot predictions
plt.plot(predictions, label='predictions')

# Menambahkan legend
plt.legend()

# Menampilkan plot
plt.show()


# In[ ]:





# In[ ]:




