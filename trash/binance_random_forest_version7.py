import ccxt.async_support as ccxt
import logging
import pandas as pd
import numpy as np
import talib
import asyncio
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from tqdm import tqdm
import joblib
from binance_settting import api, secret
from decimal import Decimal
import optuna
import warnings

warnings.filterwarnings("ignore")

logging.basicConfig(filename='trading_bot.log', level=logging.ERROR)
pd.options.mode.chained_assignment = None

class TradingBot:
    def __init__(self):
        self.exchange = None
        self.balance = None
        self.symbol = 'BTC/USDT'
        self.positions = 0
        self.add_position = 0.001
        self.max_positions = 5
        self.risk_percentage = 0.01
        self.model = None
        self.best_params = None

        # Kelly criterion
        self.probability = 0.5
        self.win_ratio = None
        self.profit_ratio = 1.2

        self.param_space = {
            'n_estimators': optuna.distributions.IntUniformDistribution(100, 700),
            'max_depth': optuna.distributions.IntUniformDistribution(5, 100),
            'min_samples_split': optuna.distributions.IntUniformDistribution(2, 10),
            'min_samples_leaf': optuna.distributions.IntUniformDistribution(1, 5),
            'max_features': optuna.distributions.CategoricalDistribution(['sqrt', 'log2']),
            'bootstrap': optuna.distributions.CategoricalDistribution([True, False])
        }

    def kelly_criterion(self, probability, win_ratio, profit_ratio):
        return (win_ratio * (profit_ratio + 1) - 1) / profit_ratio

    async def load_model(self):
        try:
            self.model = joblib.load('model.pkl')
            print('Pretrained model loaded.')
            self.best_params = self.model.get_params()
        except FileNotFoundError:
            print('Pretrained model not found. Initializing with None.')
            self.model = None

    async def initialize_exchange(self):
        self.exchange = ccxt.binance({
            'apiKey': api,
            'secret': secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',
                'adjustForTimeDifference': True,
                'test': True
            }
        })
        self.exchange.set_sandbox_mode(True)
        await self.exchange.load_markets()

    async def get_balance(self):
        self.balance = await self.exchange.fetch_balance()
        return self.balance['USDT']['free']

    async def fetch_data(self, timeframe, limit):
        klines = await self.exchange.fetch_ohlcv(self.symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df

    async def preprocess_data(self, df):
        df['rsi'] = talib.RSI(df['close'], timeperiod=14)
        df['macd'], _, _ = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
        df.dropna(inplace=True)

    async def generate_features(self, df):
        features = df[['rsi', 'macd']]
        return features

    async def train_model(self, X, y):
        model = RandomForestClassifier(**self.best_params)
        model.fit(X, y)
        self.model = model

    async def optimize_parameters(self, X, y):
        def objective(trial):
            params = {p.name: trial.suggest(p) for p in self.param_space}
            model = RandomForestClassifier(**params)
            score = np.mean(cross_val_score(model, X, y, cv=3, scoring='accuracy'))
            return score

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50)
        self.best_params = study.best_params

    async def save_model(self):
        joblib.dump(self.model, 'model.pkl')
        print('Model saved.')

    async def run(self):
        await self.initialize_exchange()
        await self.load_model()
        await self.get_balance()

        while True:
            try:
                df = await self.fetch_data('1m', 100)
                await self.preprocess_data(df)
                features = await self.generate_features(df)

                if self.model is None:
                    X = features
                    y = np.random.choice([0, 1], size=X.shape[0])
                    await self.optimize_parameters(X, y)
                    await self.train_model(X, y)
                    await self.save_model()

                signal = self.model.predict(features.iloc[-1].values.reshape(1, -1))
                signal = int(signal[0])

                if signal == 1 and self.positions < self.max_positions:
                    balance = await self.get_balance()
                    position_size = Decimal(balance) * self.risk_percentage
                    qty = position_size / df['close'].iloc[-1]
                    
                    kelly_position_size = Decimal(balance) * self.kelly_criterion(self.probability, self.win_ratio, self.profit_ratio)
                    kelly_qty = kelly_position_size / df['close'].iloc[-1]
                    
                    self.exchange.create_limit_buy_order(self.symbol, min(qty, kelly_qty), df['close'].iloc[-1])
                    self.positions += 1
                    print(f'Buy order placed. Positions: {self.positions}')

                if self.positions > 0 and self.positions % self.add_position == 0:
                    balance = await self.get_balance()
                    position_size = Decimal(balance) * self.risk_percentage
                    qty = position_size / df['close'].iloc[-1]
                    
                    kelly_position_size = Decimal(balance) * self.kelly_criterion(self.probability, self.win_ratio, self.profit_ratio)
                    kelly_qty = kelly_position_size / df['close'].iloc[-1]
                    
                    self.exchange.create_limit_buy_order(self.symbol, min(qty, kelly_qty), df['close'].iloc[-1])
                    self.positions += 1
                    print(f'Additional buy order placed. Positions: {self.positions}')

                await asyncio.sleep(0.5)  
            except Exception as e:
                logging.error(str(e))

if __name__ == '__main__':

    asyncio.get_event_loop().run_until_complete(TradingBot().run())
