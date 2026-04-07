import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import time
import telebot
import talib
import os
import sys
from dotenv import load_dotenv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from gym import Env
from gym.spaces import Box, Discrete
from transformers import pipeline
import torch
import requests
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TimeDistributed, Conv1D, MaxPooling1D, Flatten, LSTM, Dense, Dropout
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
import optuna
import logging
from tensorflow.keras.optimizers import Adam

# Load environment variables
TELEGRAM_TOKEN = "7628629022:AAFqs1BYdL1lNfFIA5oIAnXb3yBQf4u6xYQ"
CHAT_ID = "6901834585"

bot = telebot.TeleBot(TELEGRAM_TOKEN)

# Connect to MetaTrader 5
if not mt5.initialize():
    print("❌ MT5 connection failed!")
    mt5.shutdown()
    sys.exit()

logging.basicConfig(filename='trading_bot.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Transformer-based Sentiment Analyzer
sentiment_pipeline = pipeline("sentiment-analysis")

def get_news_sentiment():
    """Fetch sentiment from news headlines"""
    url = "https://newsapi.org/v2/top-headlines"
    params = {'apiKey': 'f7c21e43c2a441c89c6c5c2d41cddbe8', 'country': 'us'}
    response = requests.get(url, params=params)
    news_data = response.json()

    if news_data['status'] == 'ok' and news_data['articles']:
        news_article = news_data['articles'][0]['title']
    else:
        news_article = "Market sentiment is neutral."

    sentiment_result = sentiment_pipeline(news_article)[0]
    return "BUY" if sentiment_result['label'] == "POSITIVE" else "SELL" if sentiment_result['label'] == "NEGATIVE" else "HOLD"

# Risk Management Functions (pozitsiya hajmini hisoblash kerak emas)
def calculate_position_size(account_balance, risk_percentage, stop_loss_pips, pip_value=10):
    """Optimal pozitsiya hajmini hisoblash (kerak emas, lekin saqlab qoldim)."""
    risk_amount = account_balance * (risk_percentage / 100)
    lot_size = risk_amount / (stop_loss_pips * pip_value)
    return round(lot_size, 2)

class TradingEnv(Env):
    def __init__(self, symbol="XAUUSD", timeframe=mt5.TIMEFRAME_H1, window_size=50):
        super(TradingEnv, self).__init__()
        self.symbol = symbol
        self.timeframe = timeframe
        self.trend_window = 20
        self.window_size = window_size
        self.action_space = Discrete(3)  # 0 - Hold, 1 - Buy, 2 - Sell
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.window_size, 11), dtype=np.float32) # o'zgartirildi

    def reset(self):
        self.data = self.get_market_data()
        self.current_step = self.window_size
        return self.data.iloc[self.current_step - self.window_size:self.current_step].astype(np.float32).values

    def step(self, action):
        self.current_step += 1
        if self.current_step >= len(self.data):
            done = True
            reward = 0
        else:
            done = False
            reward = self.calculate_reward(action)

        obs = self.data.iloc[self.current_step - self.window_size:self.current_step].astype(np.float32).values
        return obs, reward, done, {}

    def get_market_data(self):
        rates = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, 10000)
        df = pd.DataFrame(rates)
        df['EMA_50'] = talib.EMA(df['close'], timeperiod=50)
        df['EMA_200'] = talib.EMA(df['close'], timeperiod=200)
        df['MACD'], df['MACD_Signal'], _ = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
        df['RSI'] = talib.RSI(df['close'], timeperiod=14)
        df['ATR'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        df['TREND'] = talib.EMA(df['close'], timeperiod=self.trend_window)
        df['UPPER'], df['MIDDLE'], df['LOWER'] = talib.BBANDS(df['close'], timeperiod=20)
        df['SLOWK'], df['SLOWD'] = talib.STOCH(df['high'], df['low'], df['close'], fastk_period=14, slowk_period=3, slowd_period=3)
        df.bfill(inplace=True)
        df.dropna(inplace=True)
        df = df[['close', 'EMA_50', 'EMA_200', 'MACD', 'RSI', 'ATR', 'TREND', 'UPPER', 'LOWER','SLOWK','SLOWD']]
        return df

    def calculate_reward(self, action):
        # Haqiqiy savdo natijalariga asoslangan mukofot berish
        # Misol: Keyingi qadamdagi narx o'zgarishiga qarab mukofot
        if self.current_step + 1 < len(self.data):
            next_close = self.data.iloc[self.current_step + 1]['close']
            current_close = self.data.iloc[self.current_step]['close']
            if action == 1:  # Buy
                return next_close - current_close
            elif action == 2:  # Sell
                return current_close - next_close
            else:  # Hold
                return 0
        else:
            return 0  # Oxirgi qadamda mukofot 0

# CNN + LSMT MODUL
class CNNLSTMFeaturesExtractor(torch.nn.Module):
    def __init__(self, observation_space):
        super(CNNLSTMFeaturesExtractor, self).__init__()
        input_shape = observation_space.shape
        self.conv1d = torch.nn.Conv1d(in_channels=input_shape[1], out_channels=64, kernel_size=3)
        self.maxpool1d = torch.nn.MaxPool1d(kernel_size=2)
        self.flatten = torch.nn.Flatten()
        self.lstm = torch.nn.LSTM(input_size=64 * ((input_shape[0] - 2) // 2), hidden_size=50, batch_first=True)
        self.dense = torch.nn.Linear(50, 3)
        self.features_dim = 3 # Qo'shildi: Chiqish o'lchami

    def forward(self, features):
        x = features.permute(0, 2, 1)
        x = torch.relu(self.conv1d(x))
        x = self.maxpool1d(x)
        x = self.flatten(x)
        x = x.unsqueeze(1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.dense(x)
        return x

# TradingEnv sinfining namunasini yaratish
env = TradingEnv()

# CNN + LSTM features extractor ni yaratish
features_extractor = CNNLSTMFeaturesExtractor(env.observation_space) # observation_space berildi

# CNN + LSTM modelini yaratish
cnn_lstm_model = Sequential()
# Birinchi CNN qatlami
cnn_lstm_model.add(TimeDistributed(Conv1D(filters=32, kernel_size=3, activation='relu'), input_shape=(50, 11, 1)))
cnn_lstm_model.add(TimeDistributed(MaxPooling1D(pool_size=3)))
cnn_lstm_model.add(TimeDistributed(Dropout(0.5)))

cnn_lstm_model.add(TimeDistributed(Flatten()))

# LSTM qatlami
cnn_lstm_model.add(LSTM(100, activation='relu'))  # Neyronlar soni biroz oshirildi
cnn_lstm_model.add(Dropout(0.5))                    # Dropout biroz kamaytirildi

# Chiqish qatlami
cnn_lstm_model.add(Dense(3, activation='softmax'))

# O'rganish tezligi biroz oshirildi
optimizer = Adam(learning_rate=0.0005)
cnn_lstm_model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Learning Rate Scheduling
def lr_schedule(epoch, lr):
    if epoch > 10:
        return lr * 0.1
    else:
        return lr

lr_scheduler = LearningRateScheduler(lr_schedule)

# Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5)

# Train LSTM model
env = TradingEnv()  # TradingEnv sinfidan namuna yaratish
env.reset() # env.reset() funksiyasini ishga tushirish
all_data = env.get_market_data()
train_data = all_data.iloc[:-2000]
test_data = all_data.iloc[-2000:]

X_train = train_data.values[:-50].reshape(-1, 50, 11)
y_train = np.array([1 if train_data['close'].iloc[i + 1] > train_data['close'].iloc[i] else 2 if train_data['close'].iloc[i + 1] < train_data['close'].iloc[i] else 0 for i in range(len(train_data) - 50)])

# Sinflar og'irliklarini hisoblash (agar sinflar nomutanosib bo'lsa)
from sklearn.utils import class_weight
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(zip(np.unique(y_train), class_weights))

cnn_lstm_model.fit(X_train, y_train, epochs=100, validation_split=0.2, callbacks=[early_stopping, lr_scheduler], class_weight=class_weight_dict)

# Modelni baholash
window_size = 50
X_test = test_data.values[:-window_size].reshape(-1, window_size, 11, 1)
y_test_list = []
for i in range(0, len(test_data) - window_size, window_size):
    if i + 2 * window_size <= len(test_data):
        next_close = test_data['close'].iloc[i + 2 * window_size - 1]
        current_close = test_data['close'].iloc[i + window_size - 1]
        if next_close > current_close:
            y_test_list.append(1)
        elif next_close < current_close:
            y_test_list.append(2)
        else:
            y_test_list.append(0)
    else:
        break # Agar to'liq keyingi oynaga ma'lumot yetmasa, to'xtatamiz
y_test = np.array(y_test_list)

print(f"X_test shape before predict: {X_test.shape}")
y_pred = cnn_lstm_model.predict(X_test)
print(f"y_pred shape after predict: {y_pred.shape}")
y_pred_classes = np.argmax(y_pred, axis=1)
print(f"y_pred_classes shape after argmax: {y_pred_classes.shape}")
print(f"y_test shape: {y_test.shape}")

from sklearn.metrics import precision_score, recall_score, f1_score

precision = precision_score(y_test, y_pred_classes, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred_classes, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred_classes, average='weighted', zero_division=0)

print(f"Test Precision: {precision}, Test Recall: {recall}, Test F1-score: {f1}")

def objective(trial):
    """Hyperparameter tuning for PPO using Optuna"""
    env = DummyVecEnv([lambda: TradingEnv()])
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    gamma = trial.suggest_uniform('gamma', 0.8, 0.999)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    model = PPO("MlpPolicy", env, learning_rate=learning_rate, gamma=gamma, batch_size=batch_size, verbose=0)
    model.learn(total_timesteps=70000)
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)
    return mean_reward

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)

best_params = study.best_params
print("Best PPO Hyperparameters:", best_params)

# Train PPO model with best hyperparameters
env = DummyVecEnv([lambda: TradingEnv()])
ppo_model = PPO("MultiInputPolicy", env, **best_params,
                policy_kwargs={"features_extractor_class": CNNLSTMFeaturesExtractor},
                verbose=1)
ppo_model.learn(total_timesteps=700000)
ppo_model.save("ppo_trading_model")

# Execute Trade with Position Sizing and Risk Management (olib tashlandi)
def execute_trade_with_risk_management(account_balance, risk_percentage, entry_price, stop_loss_pips):
    """Optimal pozitsiya hajmini hisoblash (olib tashlandi)."""
    lot_size = calculate_position_size(account_balance, risk_percentage, stop_loss_pips)
    print(f"Executing trade with lot size: {lot_size}")
    return lot_size

def send_signal_to_telegram(signal, entry, sl, tp):
    """Send trade signal to Telegram"""
    message = f"""
📌 *AI TRADING SIGNAL (LSTM + PPO + Transformer)*
🔹 *Tiker:* XAUUSD
🔹 *Signal:* {signal}
🔹 *Entry Price:* {entry:.5f}
🔹 *Stop-Loss:* {sl:.5f}
🔹 *Take-Profit:* {tp:.5f}
"""
    try:
        bot.send_message(CHAT_ID, message, parse_mode="Markdown")
        print("✅ Telegram signal sent!")
    except Exception as e:
        print(f"❌ Telegram error: {e}")

# def open_trade(symbol, trade_type, lot, price, sl, tp):
#     """Savdo pozitsiyasini ochish (olib tashlandi)."""
#     pass

# def close_trade(ticket, lot):
#     """Savdo pozitsiyasini yopish (olib tashlandi)."""
#     pass

def main():
    env = TradingEnv()
    ppo_model = PPO.load("ppo_trading_model")
    obs = env.reset()
    account_balance = 1000  # Misol uchun balans
    risk_percentage = 2  # Har bir savdoda 1% risk

    while True:
        try:
            action, _ = ppo_model.predict(obs)
            sentiment_signal = get_news_sentiment()
            signal = "BUY" if action == 1 else "SELL" if action == 2 else "HOLD"

            if sentiment_signal == "BUY" and signal != "SELL":
                signal = "BUY"
            elif sentiment_signal == "SELL" and signal != "BUY":
                signal = "SELL"

            data_h4 = TradingEnv().get_market_data()
            if data_h4 is not None:
                entry = data_h4.iloc[-1]['close']
                atr = data_h4.iloc[-1]['ATR']
                trend = data_h4.iloc[-1]['TREND']
                sl = entry - atr * 1.5
                tp = entry + atr * 2

                if signal == "BUY" and entry > trend:  # Trendga mos buy signali
                    send_signal_to_telegram(signal, entry, sl, tp)
                    logging.info(f"SIGNAL: BUY, Entry: {entry}, SL: {sl}, TP: {tp}")
                elif signal == "SELL" and entry < trend:  # Trendga mos sell signali
                    send_signal_to_telegram(signal, entry, sl, tp)
                    logging.info(f"SIGNAL: SELL, Entry: {entry}, SL: {sl}, TP: {tp}")
                else:
                    signal = "HOLD"  # Trendga mos kelmasa, HOLD
                    logging.info("SIGNAL: HOLD (trendga mos kelmadi).")
                    sl, tp = entry, entry

                if signal == "HOLD":
                    logging.info("Telegramga signal yuborilmadi (HOLD).")

            print("⌛ Waiting 1 hour...") # Vaqt oralig'ini H1 ga moslashtirdim
            time.sleep(3600) # 1 soat kutish

        except Exception as e:
            logging.error(f"Xatolik: {e}")
            time.sleep(60)  # Xatolikdan keyin 1 daqiqa kutish

if __name__ == "__main__":
    main()