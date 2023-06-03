import ccxt.async_support as ccxt
import logging
import pandas as pd
import numpy as np
import talib
import asyncio
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
import joblib
from binance_settting import api , secret
logging.basicConfig(filename='trading_bot.log',level=logging.ERROR)
pd.options.mode.chained_assignment = None

class TradingBot:
    def __init__(self):
        self.exchange = None
        self.balance = None
        self.symbol = 'BTC/USDT'
        self.positions = 0
        self.add_position = 1.0 
        self.max_positions = 5
        self.risk_percentage = 0.2
        self.model = self.load_model()
        self.param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 5, 10],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 5],
            'max_features': ['sqrt', 'log2'],
            'bootstrap': [True, False]
        }

    def load_model(self):
        try:
            model = joblib.load('model.pkl')
            print('Pretrained model loaded.')
            return model
        except FileNotFoundError:
            print('Pretrained model not found. Initializing with None.')
            return None 

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

    async def start_code(self):
        await self.initialize_exchange()
        self.balance = await self.exchange.fetch_balance()
        pbar = tqdm(total=len(self.param_grid), desc="Grid Search")
        while True:
            try:
                data = await self.exchange.fetch_ohlcv(self.symbol, '1d', limit=100)
                df = self.prepare_features(data)
                df['label'] = self.prepare_labels(df)
                features = df.drop(columns=['time', 'open', 'volume', 'label'])
                labels = df['label']

                if self.model is None:
                    self.model = self.train_model(features, labels, self.param_grid, pbar)
                else:
                    self.train_model_with_existing_model(features, labels)

                await self.next(data)

            except Exception as e:
                logging.error(e)
                await asyncio.sleep(0.5)

    async def next(self, data):
        price = data[-1][4]
        await self.trade(price)

    async def trade(self, price):
        account_balance = self.balance['total']['USDT']
        stop_loss_price, take_profit_price = await self.calculate_stop_loss_take_profit_prices()
        position_size = await self.calculate_position_size(price, stop_loss_price)

        if stop_loss_price is not None:
            await self.close_positions(stop_loss_price)

        if take_profit_price is not None:
            await self.close_positions(take_profit_price)

        if account_balance > 0 and self.positions < self.max_positions:
            buy_price = price * self.add_position
            if await self.buy(buy_price, position_size):
                self.positions += 1
        
    async def calculate_position_size(self, price, stop_loss_price):
        volatility = await self.calculate_volatility()
        risk_per_trade = self.calculate_risk_per_trade(volatility)
        position_size = risk_per_trade / (price - stop_loss_price)
        return position_size
    
    def calculate_risk_per_trade(self, volatility):
        account_balance = self.balance['total']['USDT']
        risk_percentage = self.risk_percentage  
        risk_per_trade = account_balance * risk_percentage * volatility
        return risk_per_trade
    
    async def calculate_volatility(self):
        data = await self.exchange.fetch_ohlcv(self.symbol, '1d', limit=20)
        prices = np.array([candle[4] for candle in data])
        highs = np.array([candle[2] for candle in data])
        lows = np.array([candle[3] for candle in data])
        atr = talib.ATR(highs, lows, prices, timeperiod=14)
        volatility = atr[-1]
        return volatility

    async def calculate_stop_loss_take_profit_prices(self):
        data = await self.exchange.fetch_ohlcv(self.symbol, '1d', limit=20)
        prices = np.array([candle[4] for candle in data])
        sma_20 = talib.SMA(prices, timeperiod=20)
        stop_loss_price = sma_20[-1]
        take_profit_price = stop_loss_price * 1.5
        return stop_loss_price, take_profit_price

    async def buy(self, price, size):
        if price is None or size is None:
            return False
        try:
            print(f"Buying {size} BTC at price {price}")
            response = await self.exchange.create_limit_buy_order(self.symbol, size, price)
            return True
        except Exception as e:
            logging.error(e)
            return False

    async def sell(self, price, size):
        if price is None or size is None:
            return False
        try:
            print(f"Selling {size} BTC at price {price}")
            response = await self.exchange.create_limit_sell_order(self.symbol, size, price)
            return True
        except Exception as e:
            logging.error(e)
            return False

    async def close_positions(self, price):
        try:
            positions = await self.exchange.fetch_open_orders(symbol=self.symbol)
            for position in positions:
                if position['side'] == 'buy':
                    size = position['amount']
                    await self.sell(price, size)
                    self.positions -= 1
        except Exception as e:
            logging.error(e)

    @staticmethod
    def prepare_features(data):
        df = pd.DataFrame(data, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
        prices = df['close']
        df['ma'] = prices.rolling(window=20).mean()
        df['rsi'] = talib.RSI(np.array(prices), timeperiod=14)
        macd, signal, hist = talib.MACD(np.array(prices), fastperiod=12, slowperiod=26, signalperiod=9)
        df['macd'] = macd
        df['signal'] = signal
        upper, middle, lower = talib.BBANDS(np.array(prices), timeperiod=20)
        df['upper_band'] = upper
        df['lower_band'] = lower
        stoch_k, stoch_d = talib.STOCH(np.array(df['high']), np.array(df['low']), np.array(prices), fastk_period=14,
                                       slowk_period=3, slowd_period=3)
        df['stoch_k'] = stoch_k
        df['stoch_d'] = stoch_d
        return df.dropna()

    @staticmethod
    def prepare_labels(df):
        df['label'] = np.where(df['close'].shift(-1) > df['close'], 1, -1)
        return df['label'].astype(int).tolist()

    def train_model(self, features, labels, param_grid ,pbar):
        rf_model = RandomForestClassifier()
        grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5)

        def run_grid_search():
            grid_search.fit(features, labels)
            pbar.update()

        run_grid_search()

        print("Best Parameters:", grid_search.best_params_)
        print("Best Score:", grid_search.best_score_)

        best_params = grid_search.best_params_
        model = RandomForestClassifier(**best_params)
        model.fit(features, labels)
        self.model = model
        joblib.dump(self.model, 'model.pkl')
        print('Model saved as model.pkl')

        return self.model

    def train_model_with_existing_model(self, features, labels):
        self.model.fit(features, labels)

if __name__ == "__main__":
    bot = TradingBot()
    asyncio.run(bot.start_code())