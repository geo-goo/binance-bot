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

logging.basicConfig(filename='trading_bot.log', level=logging.ERROR)
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
        self.model = None
        self.param_space = {
            'n_estimators': optuna.distributions.IntUniformDistribution(100, 300),
            'max_depth': optuna.distributions.IntUniformDistribution(5, 10),
            'min_samples_split': optuna.distributions.IntUniformDistribution(2, 10),
            'min_samples_leaf': optuna.distributions.IntUniformDistribution(1, 5),
            'max_features': optuna.distributions.CategoricalDistribution(['sqrt', 'log2']),
            'bootstrap': optuna.distributions.CategoricalDistribution([True, False])
        }

    async def load_model(self):
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
        self.model = await self.load_model()
        pbar = tqdm(total=1, desc="Optuna Search")

        def objective(trial):
            model = RandomForestClassifier(**trial.params)
            score = np.mean([cross_val_score(model, features, labels, cv=3) for _ in range(3)])
            return score

        data = await self.exchange.fetch_ohlcv(self.symbol, '1d', limit=100)
        df = self.prepare_features(data)
        df['label'] = self.prepare_labels(df)
        features = df.drop(columns=['time', 'open', 'volume', 'label'])
        labels = df['label']
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=1)
        best_params = study.best_params
        self.model = RandomForestClassifier(**best_params)
        self.train_model(features, labels, pbar)
        data = await self.exchange.fetch_ohlcv(self.symbol, '1d', limit=100)
        await self.next(data)

    def train_model(self, features, labels, pbar):
        self.model.fit(features, labels)
        # joblib.dump(self.model, 'model.pkl')
        pbar.update(1)

    async def next(self, data): 
        price = data[-1][4]
        features = self.prepare_features(data)
        labels = self.prepare_labels(features)
        await self.trade(price, features, labels)

    def prepare_features(self, data):
        df = pd.DataFrame(data, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
        
        # 定义要添加的技术指标列表
        indicators = [
            talib.RSI(df['close'],timeperiod=9) # RSI
            # talib.MACD(df['close'])[0],  # MACD
            # talib.OBV(df['close'], df['volume']),  # OBV
            # talib.SMA(df['close'], timeperiod=10),  # 移动平均线
            # talib.BBANDS(df['close'], timeperiod=20)[0],  # 布林带
            # talib.STOCH(df['high'], df['low'], df['close'])[0],  # 随机指标
            # talib.ATR(df['high'], df['low'], df['close']),  # 平均真实范围
            # talib.EMA(df['close']),  # 指数移动平均线
            # talib.MACD(df['close'])[2],  # 移动平均线收敛/发散指标
            # talib.BBANDS(df['close'], timeperiod=20)[1],  # 均线多空指标
            # talib.CCI(df['high'], df['low'], df['close']),  # 顺势指标
            # talib.MOM(df['close']),  # 动量指标
            # talib.PPO(df['close']),  # 价格振荡器
            # talib.TEMA(df['close']),  # 三重指数平滑平均线
            # talib.ERI(df['close']),  # 能量潮汐指标
            # talib.ULTOSC(df['high'], df['low'], df['close']),  # 乌龙指标
            # talib.PRICECHAN(df['high'], df['low'], df['close']),  # 价格通道指标
            # talib.ROC(df['close']),  # 价格变化百分比
            # talib.PVT(df['close'], df['volume']),  # 价格和成交量趋势指标
            # talib.RSI(df['close'],timeperiod=14),  # 顺势强弱指标
            # talib.DEMA(df['close']),  # 三重指数平滑移动平均线
            # talib.TRIX(df['close']),  # 三重指数平滑振荡器
            # talib.TRIMA(df['close']),  # 三重指数平滑相对强弱指数
            # talib.KAMA(df['close']),  # 三重指数平滑移动平均线
            # talib.MAMA(df['close'])[0],  # 三重指数平滑移动平均线趋势指标
            # talib.FAMA(df['close'])[0],  # 三重指数平滑移动平均线趋势线
            # talib.HT_DCPERIOD(df['close']),  # 三重指数平滑移动均线趋势强度指标
            # talib.HT_DCPHASE(df['close']),  # 三重指数平滑移动均线趋势强度线
            # talib.HT_PHASOR(df['close'])[0],  # 三重指数平滑移动均线趋势强度标
            # talib.HT_SINE(df['close'])[0],  # 三重指数平滑移动平均线趋势强度指标
            # talib.HT_TRENDMODE(df['close']),  # 三重指数平滑移动平均线趋势强度指标
            # talib.LINEARREG(df['close']),  # 三重指数平滑移动平均线趋势强度指标
            # talib.LINEARREG_ANGLE(df['close']),  # 三重指数平滑移动平均线趋势强度指标
            # talib.LINEARREG_INTERCEPT(df['close']),  # 三重指数平滑移动平均线趋势强度指标
            # talib.LINEARREG_SLOPE(df['close']),  # 三重指数平滑移动平均线趋势强度指标
            # talib.MIDPOINT(df['close']),  # 三重指数平滑移动平均线趋势强度指标
            # talib.MIDPRICE(df['high'], df['low']),  # 三重指数平滑移动平均线趋势强度指标
            # talib.SAR(df['high'], df['low']),  # 三重指数平滑移动平均线趋势强度指标
            # talib.SAREXT(df['high'], df['low']),  # 三重指数平滑移动平均线趋势强度指标
            # talib.SMA(df['close']),  # 三重指数平滑移动平均线趋势强度指标
            # talib.T3(df['close']),  # 三重指数平滑移动平均线趋势强度指标
            # talib.TRANGE(df['high'], df['low'], df['close']),  # 三重指数平滑移动平均线趋势强度指标
            # talib.TSF(df['close']),  # 三重指数平滑移动平均线趋势强度指标
            # talib.TYPPRICE(df['high'], df['low'], df['close']),  # 三重指数平滑移动平均线趋势强度指标
            # talib.VAR(df['close']),  # 三重指数平滑移动平均线趋势强度指标
            # talib.WCLPRICE(df['high'], df['low'], df['close']),  # 三重指数平滑移动平均线趋势强度指标
            # talib.WMA(df['close']),  # 三重指数平滑移动平均线趋势强度指标
            # talib.AD(df['high'], df['low'], df['close'], df['volume']),  # 三重指数平滑移动平均线趋势强度指标
            # talib.ADOSC(df['high'], df['low'], df['close'], df['volume']),  # 三重指数平滑移动平均线趋势强度指标
            # talib.ADXR(df['high'], df['low'], df['close']),  # 三重指数平滑移动平均线趋势强度指标
            # talib.APO(df['close']),  # 三重指数平滑移动平均线趋势强度指标
            # talib.AROON(df['high'], df['low']),  # 三重指数平滑移动平均线趋势强度指标
            # talib.AROONOSC(df['high'], df['low']),  # 三重指数平滑移动平均线趋势强度指标
            # talib.BOP(df['open'], df['high'], df['low'], df['close']),  # 三重指数平滑移动平均线趋势强度指标
            # talib.CMO(df['close']),  # 三重指数平滑移动平均线趋势强度指标
            # talib.DX(df['high'], df['low'], df['close']),  # 三重指数平滑移动平均线趋势强度指标
            # talib.MFI(df['high'], df['low'], df['close'], df['volume']),  # 三重指数平滑移动平均线趋势强度指标
            # talib.MINUS_DI(df['high'], df['low'], df['close']),  # 三重指数平滑移动平均线趋势强度指标
            # talib.MINUS_DM(df['high'], df['low']),  # 三重指数平滑移动平均线趋势强度指标
            # talib.PLUS_DI(df['high'], df['low'], df['close']),  # 三重指数平滑移动平均线趋势强度指标
            # talib.PLUS_DM(df['high'], df['low']),  # 三重指数平滑移动平均线趋势强度指标
            # talib.PPO(df['close']),  # 三重指数平滑移动平均线趋势强度指标
            # talib.ROC(df['close']),  # 三重指数平滑移动平均线趋势强度指标
            # talib.ROCP(df['close']),  # 三重指数平滑移动平均线趋势强度指标
            # talib.ROCR(df['close']),  # 三重指数平滑移动平均线趋势强度指标
            # talib.ROCR100(df['close']),  # 三重指数平滑移动平均线趋势强度指标
            # talib.RSI(df['close']),  # 三重指数平滑移动平均线趋势强度指标
            # talib.TRIX(df['close']),  # 三重指数平滑移动平均线趋势强度指标
            # talib.ULTOSC(df['high'], df['low'], df['close']),  # 三重指数平滑移动平均线趋势强度指标
            # talib.WILLR(df['high'], df['low'], df['close']),  # 三重指数平滑移动平均线趋势强度指标
        ]
        
        # 将指标添加到数据框中
        for i, indicator in enumerate(indicators):
            column_name = f'indicator_{i+1}'
            df[column_name] = indicator
        
        return df.dropna()

    def prepare_labels(self, df):
        df['return'] = df['close'].pct_change()
        df['label'] = np.where(df['return'] > 0, 1, -1)
        df.dropna(inplace=True)
        return df['label']

    async def buy(self, price):
        if self.positions < self.max_positions:
            quantity = self.balance['USDT']['free'] * self.risk_percentage / price
            if quantity >= 0.001:
                await self.exchange.create_limit_buy_order(self.symbol, quantity, price)
                self.positions += 1
                print(f"Buy order executed at price {price}.")
            else:
                print("Insufficient funds to execute buy order.")
        else:
            print("Maximum positions reached. Cannot execute buy order.")

    async def sell(self, price):
        if self.positions > 0:
            quantity = self.balance['BTC']['free'] * self.risk_percentage
            if quantity >= 0.001:
                await self.exchange.create_limit_sell_order(self.symbol, quantity, price)
                self.positions -= 1
                print(f"Sell order executed at price {price}.")
            else:
                print("Insufficient funds to execute sell order.")
        else:
            print("No positions to sell. Cannot execute sell order.")

    async def hold(self):
        print("Holding position. No action taken.")

    async def trade(self, price, features, labels):
        try :
            features = features.drop(columns=['open', 'high', 'low', 'close', 'volume'])
            prediction = self.model.predict(features[-1])
            if prediction == 1:
                await self.buy(price)
            elif prediction == -1:
                await self.sell(price)
            elif prediction == 0:
                await self.hold()
            else:
                print("Invalid prediction value:", prediction)
        except Exception as e:
            print("Prediction Error:", str(e))
        
    async def run(self):
        await self.start_code()


if __name__ == '__main__':
    bot = TradingBot()
    asyncio.run(bot.run())
