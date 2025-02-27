from __future__ import annotations

import datetime
from multiprocessing.sharedctypes import Value

import numpy as np
import pandas as pd
from stockstats import StockDataFrame as Sdf
import talib

from finrl import config
from finrl.config import INDICATORS  # Import the INDICATORS dictionary from finrl.config

from finrl.meta.preprocessor.yahoodownloader import YahooDownloader


def load_dataset(*, file_name: str) -> pd.DataFrame:
    """
    load csv dataset from path
    :return: (df) pandas dataframe
    """
    # _data = pd.read_csv(f"{config.DATASET_DIR}/{file_name}")
    _data = pd.read_csv(file_name)
    return _data


def data_split(df, start, end, target_date_col="date"):
    """
    split the dataset into training or testing using date
    :param data: (df) pandas dataframe, start, end
    :return: (df) pandas dataframe
    """
    data = df[(df[target_date_col] >= start) & (df[target_date_col] < end)]
    data = data.sort_values([target_date_col, "tic"], ignore_index=True)
    data.index = data[target_date_col].factorize()[0]
    return data


def convert_to_datetime(time):
    time_fmt = "%Y-%m-%dT%H:%M:%S"
    if isinstance(time, str):
        return datetime.datetime.strptime(time, time_fmt)


class FeatureEngineer:
    """Provides methods for preprocessing the stock price data

    Attributes
    ----------
        use_technical_indicator : boolean
            we technical indicator or not
        tech_indicator_list : list
            a list of technical indicator names (modified from neofinrl_config.py)
        use_turbulence : boolean
            use turbulence index or not
        user_defined_feature:boolean
            use user defined features or not

    Methods
    -------
    preprocess_data()
        main method to do the feature engineering

    """

    def __init__(
        self,
        use_technical_indicator=True,
        tech_indicator_list=None,  # Set default to None
        use_vix=False,
        use_turbulence=False,
        user_defined_feature=True,
    ):
        self.use_technical_indicator = use_technical_indicator
        # If a custom tech_indicator_list is provided, use it; otherwise, use the default from config
        self.tech_indicator_list = tech_indicator_list if tech_indicator_list is not None else config.INDICATORS
        self.use_vix = use_vix
        self.use_turbulence = use_turbulence
        self.user_defined_feature = user_defined_feature
    
    # def __init__(
    #     self,
    #     use_technical_indicator=True,
    #     tech_indicator_list=config.INDICATORS,
    #     use_vix=False,
    #     use_turbulence=False,
    #     user_defined_feature=True,
    # ):
    #     self.use_technical_indicator = use_technical_indicator
    #     self.tech_indicator_list = tech_indicator_list
    #     self.use_vix = use_vix
    #     self.use_turbulence = use_turbulence
    #     self.user_defined_feature = user_defined_feature

    def preprocess_data(self, df):
        """main method to do the feature engineering
        @:param config: source dataframe
        @:return: a DataMatrices object
        """
        # clean data
        df = self.clean_data(df)

        # # add technical indicators using stockstats
        # if self.use_technical_indicator:
        #     df = self.add_technical_indicator(df)
        #     print("Successfully added technical indicators")

        # add technical indicators using talib
        if self.use_technical_indicator:
            df = self.add_technical_indicator(df)
            print("Successfully added technical indicators using talib")


        # add vix for multiple stock
        if self.use_vix:
            df = self.add_vix(df)
            print("Successfully added vix")

        # add turbulence index for multiple stock
        if self.use_turbulence:
            df = self.add_turbulence(df)
            print("Successfully added turbulence index")

        # add user defined feature
        if self.user_defined_feature:
            df = self.add_user_defined_feature(df)
            print("Successfully added user defined features")

        # fill the missing values at the beginning and the end
        df = df.ffill().bfill()
        return df

    def clean_data(self, data):
        """
        clean the raw data
        deal with missing values
        reasons: stocks could be delisted, not incorporated at the time step
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        df = df.sort_values(["date", "tic"], ignore_index=True)
        df.index = df.date.factorize()[0]
        merged_closes = df.pivot_table(index="date", columns="tic", values="close")
        merged_closes = merged_closes.dropna(axis=1)
        tics = merged_closes.columns
        df = df[df.tic.isin(tics)]
        # df = data.copy()
        # list_ticker = df["tic"].unique().tolist()
        # only apply to daily level data, need to fix for minute level
        # list_date = list(pd.date_range(df['date'].min(),df['date'].max()).astype(str))
        # combination = list(itertools.product(list_date,list_ticker))

        # df_full = pd.DataFrame(combination,columns=["date","tic"]).merge(df,on=["date","tic"],how="left")
        # df_full = df_full[df_full['date'].isin(df['date'])]
        # df_full = df_full.sort_values(['date','tic'])
        # df_full = df_full.fillna(0)
        return df

    def add_technical_indicator(self, data):
        df = data.copy()
        df = df.sort_values(by=["tic", "date"])
        unique_tickers = df['tic'].unique()

        for indicator_name in INDICATORS:
            indicator_df = pd.DataFrame()
            for ticker in unique_tickers:
                ticker_data = df[df['tic'] == ticker]
                if indicator_name in INDICATORS:
                    indicator_value = INDICATORS[indicator_name](ticker_data)
                else:
                    print(f"Indicator {indicator_name} is not recognized.")
                    indicator_value = None  # or handle it in another appropriate way

                temp_df = pd.DataFrame({
                    'tic': ticker,
                    'date': ticker_data['date'],
                    indicator_name: indicator_value
                })
                indicator_df = pd.concat([indicator_df, temp_df], ignore_index=True)

            df = df.merge(indicator_df, on=['tic', 'date'], how='left')

        df = df.sort_values(by=['date', 'tic'])
        return df

    
    # def add_technical_indicator(self, data):
    #     """
    #     calculate technical indicators
    #     use stockstats package to add technical inidactors
    #     :param data: (df) pandas dataframe
    #     :return: (df) pandas dataframe
    #     """
    #     df = data.copy()
    #     df = df.sort_values(by=["tic", "date"])
    #     stock = Sdf.retype(df.copy())
    #     unique_ticker = stock.tic.unique()

    #     for indicator in self.tech_indicator_list:
    #         indicator_df = pd.DataFrame()
    #         for i in range(len(unique_ticker)):
    #             try:
    #                 temp_indicator = stock[stock.tic == unique_ticker[i]][indicator]
    #                 temp_indicator = pd.DataFrame(temp_indicator)
    #                 temp_indicator["tic"] = unique_ticker[i]
    #                 temp_indicator["date"] = df[df.tic == unique_ticker[i]][
    #                     "date"
    #                 ].to_list()
    #                 # indicator_df = indicator_df.append(
    #                 #     temp_indicator, ignore_index=True
    #                 # )
    #                 indicator_df = pd.concat(
    #                     [indicator_df, temp_indicator], axis=0, ignore_index=True
    #                 )
    #             except Exception as e:
    #                 print(e)
    #         df = df.merge(
    #             indicator_df[["tic", "date", indicator]], on=["tic", "date"], how="left"
    #         )
    #     df = df.sort_values(by=["date", "tic"])
    #     return df
        # df = data.set_index(['date','tic']).sort_index()
        # df = df.join(df.groupby(level=0, group_keys=False).apply(lambda x, y: Sdf.retype(x)[y], y=self.tech_indicator_list))
        # return df.reset_index()

    def add_user_defined_feature(self, data):
        """
        Add user defined features.
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
    
        # Calculate 'mom_s1' and its derivative
        mom_s1_features = self.calculate_user_defined_feature(df)
    
        # Merge the new features into the original dataframe
        df = df.merge(mom_s1_features, left_index=True, right_index=True)
    
        # Uncomment and include any other feature calculations as needed
        # df["daily_return"] = df.close.pct_change(1)
        # df['return_lag_1'] = df.close.pct_change(2)
        # df['return_lag_2'] = df.close.pct_change(3)
    
        return df

    # def calculate_user_defined_feature(self, data):
    #     df = data.copy()
    #     df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    #     df['price_diff'] = df['typical_price'].diff().abs()
        
    #     # Calculate current price difference for each row
    # #    df['current_price_diff'] = df['typical_price'].diff().abs()
    
    #     # Calculate the 99th percentile (0.99 quantile) for absolute difference in each rolling window
    #     df['coeff_e_diff'] = df['price_diff'].rolling(window=90).apply(lambda x: x.quantile(0.99)) * 1
    
    #     # Calculate 'mom_s1' for each row, handling division by zero
    #     df['mom_s1'] = df.apply(lambda row: row['price_diff'] / row['coeff_e_diff'] 
    #                             if row['coeff_e_diff'] != 0 else None, axis=1)
    #     return df['mom_s1']


    def calculate_user_defined_feature(self, data):
        df = data.copy()
    
        # Replace suspicious data with NaN (customize this according to your data's characteristics)
        df[['high', 'low', 'close']] = df[['high', 'low', 'close']].replace(0, np.nan)
        # Add any other conditions for suspicious data here
    
        # Forward fill to replace NaNs with the last valid observation
        df[['high', 'low', 'close']] = df[['high', 'low', 'close']].fillna(method='ffill')
    
        # Calculate typical price and price difference
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['price_diff'] = df['typical_price'].diff().abs()
    
        # Calculate the 99th percentile (0.99 quantile) in each rolling window
        df['coeff_e_diff'] = df['price_diff'].rolling(window=14).apply(lambda x: x.quantile(0.99)) * 1
    
        # Calculate 'mom_s1' for each row, handling division by zero
        df['mom_s1'] = df.apply(lambda row: row['price_diff'] / row['coeff_e_diff'] if row['coeff_e_diff'] > 0 else None, axis=1)

        # Calculate the derivative of 'mom_s1'
        df['mom_s1_derivative'] = df['mom_s1'].diff()
    
        return df[['mom_s1', 'mom_s1_derivative']]
    
    
    
    def add_vix(self, data):
        """
        add vix from yahoo finance
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        df_vix = YahooDownloader(
            start_date=df.date.min(), end_date=df.date.max(), ticker_list=["^VIX"]
        ).fetch_data()
        vix = df_vix[["date", "close"]]
        vix.columns = ["date", "vix"]

        df = df.merge(vix, on="date")
        df = df.sort_values(["date", "tic"]).reset_index(drop=True)
        return df

    def add_turbulence(self, data):
        """
        add turbulence index from a precalcualted dataframe
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        turbulence_index = self.calculate_turbulence(df)
        df = df.merge(turbulence_index, on="date")
        df = df.sort_values(["date", "tic"]).reset_index(drop=True)
        return df

    def calculate_turbulence(self, data):
        """calculate turbulence index based on dow 30"""
        # can add other market assets
        df = data.copy()
        df_price_pivot = df.pivot(index="date", columns="tic", values="close")
        # use returns to calculate turbulence
        df_price_pivot = df_price_pivot.pct_change()

        unique_date = df.date.unique()
        # start after a year
        start = 252
        turbulence_index = [0] * start
        # turbulence_index = [0]
        count = 0
        for i in range(start, len(unique_date)):
            current_price = df_price_pivot[df_price_pivot.index == unique_date[i]]
            # use one year rolling window to calcualte covariance
            hist_price = df_price_pivot[
                (df_price_pivot.index < unique_date[i])
                & (df_price_pivot.index >= unique_date[i - 252])
            ]
            # Drop tickers which has number missing values more than the "oldest" ticker
            filtered_hist_price = hist_price.iloc[
                hist_price.isna().sum().min() :
            ].dropna(axis=1)

            cov_temp = filtered_hist_price.cov()
            current_temp = current_price[[x for x in filtered_hist_price]] - np.mean(
                filtered_hist_price, axis=0
            )
            # cov_temp = hist_price.cov()
            # current_temp=(current_price - np.mean(hist_price,axis=0))

            temp = current_temp.values.dot(np.linalg.pinv(cov_temp)).dot(
                current_temp.values.T
            )
            if temp > 0:
                count += 1
                if count > 2:
                    turbulence_temp = temp[0][0]
                else:
                    # avoid large outlier because of the calculation just begins
                    turbulence_temp = 0
            else:
                turbulence_temp = 0
            turbulence_index.append(turbulence_temp)
        try:
            turbulence_index = pd.DataFrame(
                {"date": df_price_pivot.index, "turbulence": turbulence_index}
            )
        except ValueError:
            raise Exception("Turbulence information could not be added.")
        return turbulence_index
