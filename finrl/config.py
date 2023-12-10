# directory
from __future__ import annotations
import numpy as np
import pandas as pd
import talib

DATA_SAVE_DIR = "datasets"
TRAINED_MODEL_DIR = "trained_models"
TENSORBOARD_LOG_DIR = "tensorboard_log"
RESULTS_DIR = "results"

# date format: '%Y-%m-%d'
TRAIN_START_DATE = "2014-01-06"  # bug fix: set Monday right, start date set 2014-01-01 ValueError: all the input array dimensions for the concatenation axis must match exactly, but along dimension 0, the array at index 0 has size 1658 and the array at index 1 has size 1657
TRAIN_END_DATE = "2020-07-31"

TEST_START_DATE = "2020-08-01"
TEST_END_DATE = "2021-10-01"

TRADE_START_DATE = "2021-11-01"
TRADE_END_DATE = "2021-12-01"

# stockstats technical indicator column names
# check https://pypi.org/project/stockstats/ for different names
# INDICATORS = [
#     "macd",
#     "boll_ub",
#     "boll_lb",
#     "rsi_30",
#     "cci_30",
#     "dx_30",
#     "close_30_sma",
#     "close_60_sma",
#     "aroon",
#     "kdjd",
#     "kdjj",
#     "kdjk",
#     "mad",
#     "trix",
#     "ftr_20",
#     "inertia",
#     "kst",
#     "ppo",
#     "rsv",
#     "rvgi",
#     "tema"
# ]

# INDICATORS = {
#     "macd": lambda df: talib.MACD(df['close'])[0],  # MACD
#     "boll_ub": lambda df: talib.BBANDS(df['close'])[0],  # Upper Bollinger Bands
#     "boll_lb": lambda df: talib.BBANDS(df['close'])[2],  # Lower Bollinger Bands
#     "rsi_30": lambda df: talib.RSI(df['close'], timeperiod=30),
#     # "cci_30": lambda df: talib.CCI(df['high'], df['low'], df['close'], timeperiod=30),
#     # "dx_30": lambda df: talib.DX(df['high'], df['low'], df['close'], timeperiod=30),
#     "close_30_sma": lambda df: talib.SMA(df['close'], timeperiod=30),
#     "close_60_sma": lambda df: talib.SMA(df['close'], timeperiod=60),
#     # "aroon": lambda df: talib.AROON(df['high'], df['low'])[0],  # Aroon up
#     # "kdjk": lambda df: talib.STOCH(df['high'], df['low'], df['close'])[0],  # Stochastic %K
#     # "kdjd": lambda df: talib.STOCH(df['high'], df['low'], df['close'])[1],  # Stochastic %D
#     # "kdjj": lambda df: 3 * df['kdjk'] - 2 * df['kdjd'],  # J line not directly available in talib
#     # "trix": lambda df: talib.TRIX(df['close']),
#     # Add more indicators as needed

#     # New indicators
#     "KAMA": lambda df: talib.KAMA(df['close']),
#     "Stochastic": lambda df: talib.STOCH(df['high'], df['low'], df['close'])[0],  # Stochastic %K
#     "AROON": lambda df: talib.AROON(df['high'], df['low'])[0],  # Aroon up
#     "UltimateOscillator": lambda df: talib.ULTOSC(df['high'], df['low'], df['close']),
#     "ADXR": lambda df: talib.ADXR(df['high'], df['low'], df['close']),
#     "WilliamsR": lambda df: talib.WILLR(df['high'], df['low'], df['close']),
#     "ROC": lambda df: talib.ROC(df['close']),
# }

# To calculate an indicator:
# df['macd'] = INDICATORS['macd'](df)

INDICATORS = {
    # Volume Indicators
    "mfi": lambda df: talib.MFI(df['high'], df['low'], df['close'], df['volume']),
    "obv": lambda df: talib.OBV(df['close'], df['volume']),
    # Chaikin Money Flow (CMF), Force Index (FI), Ease of Movement (EoM, EMV), 
    # Volume-price Trend (VPT), Negative Volume Index (NVI) might require custom implementations

    # Volatility Indicators
    "atr": lambda df: talib.ATR(df['high'], df['low'], df['close'], timeperiod=14),
    "boll_ub": lambda df: talib.BBANDS(df['close'], timeperiod=20)[0],
    "boll_lb": lambda df: talib.BBANDS(df['close'], timeperiod=20)[2],
    # Keltner Channel (KC), Donchian Channel (DC), Ulcer Index (UI) might require custom implementations

    # Trend Indicators
    "sma": lambda df: talib.SMA(df['close'], timeperiod=30),
    "ema": lambda df: talib.EMA(df['close'], timeperiod=30),
    "wma": lambda df: talib.WMA(df['close'], timeperiod=30),
    "macd": lambda df: talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)[0],
    "adx": lambda df: talib.ADX(df['high'], df['low'], df['close'], timeperiod=14),
    # Vortex Indicator (VI), Trix (TRIX), Mass Index (MI), Commodity Channel Index (CCI), 
    # Detrended Price Oscillator (DPO), KST Oscillator (KST), Ichimoku Kinkō Hyō (Ichimoku),
    # Parabolic Stop And Reverse (Parabolic SAR), Schaff Trend Cycle (STC) might require custom implementations

    # Momentum Indicators
    "rsi": lambda df: talib.RSI(df['close'], timeperiod=14),
    "stochrsi": lambda df: talib.STOCHRSI(df['close'], timeperiod=14)[0],
    "uo": lambda df: talib.ULTOSC(df['high'], df['low'], df['close'], timeperiod1=7, timeperiod2=14, timeperiod3=28),
    "stoch": lambda df: talib.STOCH(df['high'], df['low'], df['close'], fastk_period=14, slowk_period=3, slowd_period=3)[0],
    "willr": lambda df: talib.WILLR(df['high'], df['low'], df['close'], timeperiod=14),
    "ao": lambda df: talib.APO(df['close'], fastperiod=5, slowperiod=34),
    # Kaufman’s Adaptive Moving Average (KAMA), Rate of Change (ROC), Percentage Price Oscillator (PPO),
    # Percentage Volume Oscillator (PVO) might require custom implementations

    # Other Indicators
    "daily_return": lambda df: df['close'].pct_change(),
    "daily_log_return": lambda df: np.log(df['close'] / df['close'].shift(1)),
    "cumulative_return": lambda df: (1 + df['close'].pct_change()).cumprod()
}


# Model Parameters
A2C_PARAMS = {"n_steps": 5, "ent_coef": 0.01, "learning_rate": 0.0007}
PPO_PARAMS = {
    "n_steps": 2048,
    "ent_coef": 0.01,
    "learning_rate": 0.00025,
    "batch_size": 64,
}
DDPG_PARAMS = {"batch_size": 128, "buffer_size": 50000, "learning_rate": 0.001}
TD3_PARAMS = {"batch_size": 100, "buffer_size": 1000000, "learning_rate": 0.001}
SAC_PARAMS = {
    "batch_size": 64,
    "buffer_size": 100000,
    "learning_rate": 0.0001,
    "learning_starts": 100,
    "ent_coef": "auto_0.1",
}
ERL_PARAMS = {
    "learning_rate": 3e-5,
    "batch_size": 2048,
    "gamma": 0.985,
    "seed": 312,
    "net_dimension": 512,
    "target_step": 5000,
    "eval_gap": 30,
    "eval_times": 64,  # bug fix:KeyError: 'eval_times' line 68, in get_model model.eval_times = model_kwargs["eval_times"]
}
RLlib_PARAMS = {"lr": 5e-5, "train_batch_size": 500, "gamma": 0.99}


# Possible time zones
TIME_ZONE_SHANGHAI = "Asia/Shanghai"  # Hang Seng HSI, SSE, CSI
TIME_ZONE_USEASTERN = "US/Eastern"  # Dow, Nasdaq, SP
TIME_ZONE_PARIS = "Europe/Paris"  # CAC,
TIME_ZONE_BERLIN = "Europe/Berlin"  # DAX, TECDAX, MDAX, SDAX
TIME_ZONE_JAKARTA = "Asia/Jakarta"  # LQ45
TIME_ZONE_SELFDEFINED = "xxx"  # If neither of the above is your time zone, you should define it, and set USE_TIME_ZONE_SELFDEFINED 1.
USE_TIME_ZONE_SELFDEFINED = 0  # 0 (default) or 1 (use the self defined)

# parameters for data sources
ALPACA_API_KEY = "xxx"  # your ALPACA_API_KEY
ALPACA_API_SECRET = "xxx"  # your ALPACA_API_SECRET
ALPACA_API_BASE_URL = "https://paper-api.alpaca.markets"  # alpaca url
BINANCE_BASE_URL = "https://data.binance.vision/"  # binance url
