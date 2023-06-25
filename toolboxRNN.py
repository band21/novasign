import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.metrics import mean_squared_error
import plotly.express as px

class toolboxRNN:
    df_train = pd.DataFrame()
    features = []
    target = ''
    scaler_features = MinMaxScaler(feature_range=(0, 1))
    scaler_target = MinMaxScaler(feature_range=(0, 1))
    look_back = 10
    model = Sequential()
    
    def __init__(self, train_dataset, features, target, look_back=10):
        self.df_train = train_dataset.copy()
        self.features = features
        self.target = target
        self.lool_back = look_back
        self.scaler_features.fit(train_dataset[features])
        self.scaler_target.fit(train_dataset[target].values.reshape(-1, 1))
    
    def transform_data(self, dataset=None):
        if dataset is None:
            dataset = self.df_train    
        X = []
        y = []
        for runid in dataset.runid.unique():
            df = dataset[dataset.runid == runid][self.features + [self.target]]
            df.reset_index(drop=True, inplace=True)
            y_data = df[self.target]
            y_data = self.scaler_target.transform(y_data.values.reshape(-1,1))
            for i in range(self.look_back, len(df)):
                point = df.loc[i-self.look_back:i][self.features]
                X.append(self.scaler_features.transform(point))
                y.append(y_data[i])
        return np.array(X), np.array(y)
    
    def train_model(self, X_train=None, y_train=None):
        if X_train is None or y_train is None:
            X_train, y_train = self.transform_data()
        self.model = Sequential()
        self.model.add(LSTM(10, input_shape=(X_train.shape[1], X_train.shape[2])))
        self.model.add(Dense(1))
        self.model.compile(loss='mean_squared_error', optimizer='adam')
        self.model.fit(X_train, y_train, epochs=50, batch_size=1, verbose=2)
        return self.model
    
    def evaluate(self, test_dataset, model=None):
        if model == None:
            model = self.model
        X_test, y_test = self.transform_data(test_dataset)
        y_pred = model.predict(X_test)
        y_test_unscaled = self.scaler_target.inverse_transform(y_test)
        y_pred_unsacled = self.scaler_target.inverse_transform(y_pred)
        mse_total = mean_squared_error(y_test_unscaled, y_pred_unsacled)
        mse_runs = []
        for runid in test_dataset.runid.unique():
            X_test, y_test = self.transform_data(test_dataset[test_dataset.runid == runid])
            y_pred = model.predict(X_test)
            y_test_original = test_dataset[test_dataset.runid == runid][self.target].iloc[self.look_back:]
            y_pred_unsacled = self.scaler_target.inverse_transform(y_pred)
            mse_runs.append([runid, mean_squared_error(y_test_original,y_pred_unsacled)])
        return mse_total, mse_runs
    
    def predict(self, X):
        y_pred = self.model.predict(X)
        y_pred_unsacled = self.scaler_target.inverse_transform(y_pred)
        return y_pred_unsacled
    
    def plot_test_prediction(self, test_dataset, model=None):
        if model == None:
            model = self.model
        plots = []
        for runid in test_dataset.runid.unique():
            X_test, y_test = self.transform_data(test_dataset[test_dataset.runid == runid])
            y_pred = model.predict(X_test)
            y_pred_unsacled = self.scaler_target.inverse_transform(y_pred)
            target_original_interpolated = test_dataset[test_dataset.runid == runid][self.target].reset_index(drop=True)
            target_original_interpolated.name = 'test'
            plot_data = pd.DataFrame(target_original_interpolated)
            prediction = pd.Series([None]*(len(target_original_interpolated) - len(y_pred_unsacled)) + [x[0] for x in y_pred_unsacled])
            plot_data['prediction'] = prediction
            plot_data['time'] = plot_data.index
            plot_data = plot_data.melt(id_vars='time')
            fig = px.scatter(plot_data, x='time', y='value', color='variable', title=runid)
            plots.append([runid, fig])
        return plots