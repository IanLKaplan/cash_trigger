
from enum import Enum

from IPython.core.display import Image
from dateutil.relativedelta import relativedelta
from numpy import sqrt


from datetime import datetime, timedelta
from tabulate import tabulate
from typing import List, Tuple
from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
from pandas.core.indexes.datetimes import DatetimeIndex
import numpy as np
from pathlib import Path
import tempfile

# QuantStrats was written by Ran Aroussi. See https://aroussi.com/
# See https://pypi.org/project/QuantStats/
# # https://github.com/ranaroussi/quantstats
# pip install QuantStats
import quantstats as qs

plt.style.use('seaborn-whitegrid')
pd.options.mode.chained_assignment = 'raise'


def get_market_data(file_name: str,
                    data_col: str,
                    symbols: List,
                    data_source: str,
                    start_date: datetime,
                    end_date: datetime) -> pd.DataFrame:
    """
      file_name: the file name in the temp directory that will be used to store the data
      data_col: the type of data - 'Adj Close', 'Close', 'High', 'Low', 'Open', Volume'
      symbols: a list of symbols to fetch data for
      data_source: yahoo, etc...
      start_date: the start date for the time series
      end_date: the end data for the time series
      Returns: a Pandas DataFrame containing the data.

      If a file of market data does not already exist in the temporary directory, fetch it from the
      data_source.
    """
    temp_root: str = tempfile.gettempdir() + '/'
    file_path: str = temp_root + file_name
    temp_file_path = Path(file_path)
    file_size = 0
    if temp_file_path.exists():
        file_size = temp_file_path.stat().st_size

    if file_size > 0:
        close_data = pd.read_csv(file_path, index_col='Date')
    else:
        if type(symbols) == str:
            t = list()
            t.append(symbols)
            symbols = t
        panel_data: pd.DataFrame = data.DataReader(symbols, data_source, start_date, end_date)
        close_data: pd.DataFrame = panel_data[data_col]
        close_data.to_csv(file_path)
    assert len(close_data) > 0, f'Error reading data for {symbols}'
    return close_data


trading_days = 252
trading_quarter = trading_days // 4

window_size = 200

data_source = 'yahoo'

start_date_str = '2008-03-03'
start_date: datetime = datetime.fromisoformat(start_date_str)
# The "current date"
end_date: datetime = datetime.today() - timedelta(days=1)


def convert_date(some_date):
    if type(some_date) == str:
        some_date = datetime.fromisoformat(some_date)
    elif type(some_date) == np.datetime64:
        ts = (some_date - np.datetime64('1970-01-01T00:00')) / np.timedelta64(1, 's')
        some_date = datetime.utcfromtimestamp(ts)
    return some_date


def findDateIndex(date_index: DatetimeIndex, search_date: datetime) -> int:
    '''
    In a DatetimeIndex, find the index of the date that is nearest to search_date.
    This date will either be equal to search_date or the next date that is less than
    search_date
    '''
    index: int = -1
    i = 0
    search_date = convert_date(search_date)
    date_t = datetime.today()
    for i in range(0, len(date_index)):
        date_t = convert_date(date_index[i])
        if date_t >= search_date:
            break
    if date_t > search_date:
        index = i - 1
    else:
        index = i
    return index


class RiskState(Enum):
    RISK_OFF = 0
    RISK_ON = 1


class SpyData:
    spy_close_file = 'spy_close'
    spy_etf = "SPY"
    spy_close: pd.DataFrame
    date_index: DatetimeIndex

    def __init__(self, start_date: datetime, end_date: datetime) -> object:
        """
        start_date: the start of the period for the SPY data.  One year
        before this date will be fetched to allow for a moving average
        from start_date
        end_date: the end of the period for the SPY data.
        """
        spy_start: datetime = start_date - timedelta(days=365)
        self.spy_close = get_market_data(file_name=self.spy_close_file,
                                         data_col='Close',
                                         symbols=self.spy_etf,
                                         data_source=data_source,
                                         start_date=spy_start,
                                         end_date=end_date)
        self.date_index = self.spy_close.index

    def close_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Return a section of SPY close prices from start_date to end_date
        """
        start_ix = findDateIndex(date_index=self.date_index, search_date=start_date)
        end_ix = findDateIndex(date_index=self.date_index, search_date=end_date)
        spy_close_df = pd.DataFrame()
        assert start_ix >= 0 and end_ix >= 0
        spy_close_df = self.spy_close[:][start_ix:end_ix+1].copy()
        return spy_close_df

    def avg(self, day: datetime, window: int = window_size) -> float:
        """
        :param day: the end date for the window
        :param window: the size of the window extending back from day
        :return: the average for the SPY close prices in the window
        """
        _average: float = -1.0
        end_ix = findDateIndex(date_index=self.date_index, search_date=day)
        start_ix = end_ix - (window -1)
        assert start_ix >= 0 and end_ix >= 0
        _average = self.spy_close.values[start_ix:end_ix+1].mean()
        return _average


    def moving_avg(self, start_date: datetime, end_date: datetime, window: int = window_size) -> pd.DataFrame:
        """
        Compute a moving average series
        :param start_date: the start date for the moving average series
        :param end_date: the end date for the moving average series
        :param window: the window size that extends, initially, back from start_date
        :return: a moving average series as a DataFrame. The date index will be the same
                 as the SPY time DataFrame between start_date and end_date
        """
        start_ix = findDateIndex(date_index=self.date_index, search_date=start_date)
        end_ix = findDateIndex(date_index=self.date_index, search_date=end_date)
        assert start_ix >= 0 and end_ix >= 0
        num_vals = (end_ix - start_ix) + 1
        moving_avg_a = np.zeros(num_vals)
        avg_index = self.date_index[start_ix:end_ix + 1]
        for i in range(0, num_vals):
            date_i = avg_index[i]
            mv_avg_i = self.avg(date_i)
            moving_avg_a[i] = mv_avg_i
        avg_df = pd.DataFrame(moving_avg_a)
        avg_df.index = avg_index
        avg_df.columns = [f'{self.spy_etf} {window_size}-day avg']
        return avg_df


    def risk_state(self, day: datetime) -> RiskState:
        spy_avg = self.avg(day)
        day_ix = findDateIndex(date_index=self.date_index, search_date=day)
        spy_val = float(self.spy_close.values[day_ix:day_ix+1])
        # only use full dollar price changes
        spy_avg = round(spy_avg, 0)
        spy_val = round(spy_val, 0)
        state: RiskState = RiskState.RISK_ON if spy_val > spy_avg else RiskState.RISK_OFF
        return state


spy_data = SpyData(start_date, end_date)

spy_close = spy_data.close_data(start_date, end_date)
spy_moving_avg = spy_data.moving_avg(start_date, end_date)
plot_df = pd.concat([spy_close, spy_moving_avg], axis=1)


def chooseAssetName(start: int, end: int, asset_set: pd.DataFrame) -> str:
    '''
    Choose an ETF asset in a particular range of close price values.
    The function returns the name of the asset with the highest returning
    asset in the period.
    '''
    asset_columns = asset_set.columns
    asset_name = asset_columns[0]
    if len(asset_columns) > 1:
        ret_list = []
        start_date = asset_set.index[start]
        end_date = asset_set.index[end]
        for asset in asset_set.columns:
            ts = asset_set[asset][start:end+1]
            start_val = ts[0]
            end_val = ts[-1]
            r = (end_val/start_val) - 1
            ret_list.append(r)
        ret_df = pd.DataFrame(ret_list).transpose()
        ret_df.columns = asset_set.columns
        ret_df = round(ret_df, 3)
        column = ret_df.idxmax(axis=1)[0]
        asset_name = column
    return asset_name


def find_month_periods(start_date: datetime, end_date:datetime, data: pd.DataFrame) -> pd.DataFrame:
    start_date = convert_date(start_date)
    end_date = convert_date(end_date)
    date_index = data.index
    start_ix = findDateIndex(date_index, start_date)
    end_ix = findDateIndex(date_index, end_date)
    start_l = list()
    end_l = list()
    cur_month = start_date.month
    start_l.append(start_ix)
    i = 0
    for i in range(start_ix, end_ix+1):
        date_i = convert_date(date_index[i])
        if date_i.month != cur_month:
            end_l.append(i-1)
            start_l.append(i)
            cur_month = date_i.month
    end_l.append(i)
    # if there is not a full month period, remove the last period
    if end_l[-1] - start_l[-1] < 18:
        end_l.pop()
        start_l.pop()
    start_df = pd.DataFrame(start_l)
    end_df = pd.DataFrame(end_l)
    start_date_df = pd.DataFrame(date_index[start_l])
    end_date_df = pd.DataFrame(date_index[end_l])
    periods_df = pd.concat([start_df, start_date_df, end_df, end_date_df], axis=1)
    periods_df.columns = ['start_ix', 'start_date', 'end_ix', 'end_date']
    return periods_df


def find_year_periods(data: pd.DataFrame) -> pd.DataFrame:
    date_index = data.index
    start_l = list()
    end_l = list()
    year_l = list()
    start_l.append(0)
    current_year = convert_date(date_index[0]).year
    year_l.append(current_year)
    for i in range(1, len(date_index)):
        date_i = convert_date(date_index[i])
        year_i = date_i.year
        if (year_i > current_year):
            end_l.append(i-1)
            start_l.append(i)
            year_l.append(year_i)
            current_year = year_i
    end_l.append(i)
    periods_df = pd.DataFrame(list(zip(start_l, end_l, year_l)), columns=['start_ix', 'end_ix', 'year'])
    return periods_df


def simple_return(time_series: np.array, period: int = 1) -> List :
    return list(((time_series[i]/time_series[i-period]) - 1.0 for i in range(period, len(time_series), period)))


def return_df(time_series_df: pd.DataFrame) -> pd.DataFrame:
    r_df: pd.DataFrame = pd.DataFrame()
    time_series_a: np.array = time_series_df.values
    return_l = simple_return(time_series_a, 1)
    r_df = pd.DataFrame(return_l)
    date_index = time_series_df.index
    r_df.index = date_index[1:len(date_index)]
    r_df.columns = time_series_df.columns
    return r_df


def apply_return(start_val: float, return_df: pd.DataFrame) -> np.array:
    port_a: np.array = np.zeros( return_df.shape[0] + 1)
    port_a[0] = start_val
    return_a = return_df.values
    for i in range(1, len(port_a)):
        port_a[i] = port_a[i-1] + port_a[i-1] * return_a[i-1]
    return port_a


def portfolio_return(holdings: float,
                     risk_asset: pd.DataFrame,
                     bond_asset: pd.DataFrame,
                     spy_data: SpyData,
                     start_date: datetime,
                     end_date: datetime) -> Tuple[pd.DataFrame, pd.DataFrame]:
    portfolio_a = np.zeros(0)
    portfolio_val = holdings
    name_l: List = []
    date_l: List = []
    month_periods = find_month_periods(start_date, end_date, risk_asset)
    for index, period in month_periods.iterrows():
        start_ix = period['start_ix']
        end_ix = period['end_ix']
        period_start_date: datetime = convert_date(period['start_date'])
        r_df = pd.DataFrame()
        asset_name = ''
        if spy_data.risk_state(period_start_date) == RiskState.RISK_ON:
            asset_name: str = chooseAssetName(start_ix, end_ix, risk_asset)
            risk_close_prices = pd.DataFrame(risk_asset[asset_name][start_ix:end_ix+1])
            r_df = return_df(risk_close_prices)
        else: # RISK_OFF - bonds
            asset_name: str = chooseAssetName(start_ix, end_ix, bond_asset)
            bond_close_prices = pd.DataFrame(bond_asset[asset_name][start_ix:end_ix+1])
            r_df = return_df(bond_close_prices)
        name_l.append(asset_name)
        date_l.append(period['start_date'])
        port_month_a = apply_return(portfolio_val, r_df)
        portfolio_val = port_month_a[-1]
        portfolio_a = np.append(portfolio_a, port_month_a)
    portfolio_df = pd.DataFrame(portfolio_a)
    portfolio_df.columns = ['portfolio']
    num_rows = month_periods.shape[0]
    first_row = month_periods[:][0:1]
    last_row = month_periods[:][num_rows - 1:num_rows]
    start_ix = first_row['start_ix'].values[0]
    end_ix = last_row['end_ix'].values[0]
    date_index = risk_asset.index
    portfolio_index = date_index[start_ix:end_ix + 1]
    portfolio_df.index = portfolio_index
    assets_df = pd.DataFrame(name_l)
    assets_df.columns = ['asset']
    assets_df.index = date_l
    return portfolio_df, assets_df


def adjust_time_series(ts_one_df: pd.DataFrame, ts_two_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Adjust two DataFrame time series with overlapping date indices so that they
    are the same length with the same date indices.
    """
    ts_one_index = pd.to_datetime(ts_one_df.index)
    ts_two_index = pd.to_datetime(ts_two_df.index)
        # filter the close prices
    matching_dates = ts_one_index.isin( ts_two_index )
    ts_one_adj = ts_one_df[matching_dates]
    # filter the rf_prices
    ts_one_index = pd.to_datetime(ts_one_adj.index)
    matching_dates = ts_two_index.isin(ts_one_index)
    ts_two_adj = ts_two_df[matching_dates]
    return ts_one_adj, ts_two_adj


def build_plot_data(holdings: float, portfolio_df: pd.DataFrame, spy_df: pd.DataFrame) -> pd.DataFrame:
    port_start_date = portfolio_df.index[0]
    port_start_date = convert_date(port_start_date)
    port_end_date = portfolio_df.index[-1]
    port_end_date = convert_date(port_end_date)
    spy_index = spy_df.index
    spy_start_ix = findDateIndex(spy_index, port_start_date)
    spy_end_ix = findDateIndex(spy_index, port_end_date)
    spy_df = pd.DataFrame(spy_df[:][spy_start_ix:spy_end_ix+1])
    spy_return = return_df(spy_df)
    spy_return_a = apply_return(start_val=holdings, return_df=spy_return)
    spy_port = pd.DataFrame(spy_return_a)
    spy_port.columns = ['SPY']
    spy_port.index = pd.to_datetime(spy_df.index)
    plot_df = portfolio_df.copy()
    plot_df['SPY'] = spy_port
    return plot_df


spy_start_val = float(spy_close[:].values[0])
spy_only_portfolio, t = portfolio_return(holdings=spy_start_val,
                                         risk_asset=spy_close,
                                         bond_asset=spy_close,
                                         spy_data=spy_data,
                                         start_date=start_date,
                                         end_date=end_date)

plot_df = build_plot_data(holdings=spy_start_val, portfolio_df=spy_only_portfolio, spy_df=spy_close)
plot_df.plot(grid=True, title='SPY as the only portfolio asset and SPY', figsize=(10,6))


def collapse_asset_df(asset_df: pd.DataFrame) -> pd.DataFrame:
    row = asset_df[0:1]
    cur_asset = row['asset'][0]
    collapse_df = pd.DataFrame(row)
    for index in range(1, asset_df.shape[0]):
        row = asset_df[:][index:index+1]
        row_name = row['asset'][0]
        if row_name != cur_asset:
            collapse_df = pd.concat([collapse_df, row])
            cur_asset = row_name
    if collapse_df.tail(1).index[0] != asset_df.tail(1).index[0]:
        collapse_df = pd.concat([collapse_df, asset_df.tail(1)])
    return collapse_df


holdings = 100000
cash_trigger_bonds = ['JNK', 'TLT', 'MUB']
cash_trigger_bond_file = "cash_trigger_bond_adjclose"
cash_trigger_bond_adjclose = get_market_data(file_name=cash_trigger_bond_file,
                                             data_col='Adj Close',
                                             symbols=cash_trigger_bonds,
                                             data_source=data_source,
                                             start_date=start_date,
                                             end_date=end_date)

shy_file = "shy_adjclose"
shy_ticker = 'SHY'
shy_sym = [shy_ticker]
shy_adjclose = get_market_data(file_name=shy_file,
                               data_col='Adj Close',
                               symbols=shy_sym,
                               data_source=data_source,
                               start_date=start_date,
                               end_date=end_date)

# Add SHY to the bond set
cash_trigger_bond_adjclose[shy_ticker] = shy_adjclose

spy_portfolio_df, assets_df = portfolio_return(holdings=holdings,
                                               risk_asset=spy_close,
                                               bond_asset=cash_trigger_bond_adjclose,
                                               spy_data=spy_data,
                                               start_date=start_date,
                                               end_date=end_date
                                               )


plot_df = build_plot_data(holdings=holdings, portfolio_df=spy_portfolio_df, spy_df=spy_close)


assets_collapsed_df = collapse_asset_df(assets_df)

# print(tabulate(assets_collapsed_df, headers=[*assets_collapsed_df.columns], tablefmt='fancy_grid'))


def calculate_volatility(prices: pd.DataFrame) -> pd.DataFrame:
    sd_a = np.zeros(prices.shape[1])
    for i, col in enumerate(prices.columns):
        ts = prices[col]
        ts_df = pd.DataFrame(ts)
        ts_df.columns = [col]
        ret_df = return_df(ts_df)
        sd = float(round(ret_df.std() * sqrt(trading_days) * 100, 2))
        sd_a[i] = sd
    vol_df = pd.DataFrame(sd_a).transpose()
    vol_df.columns = prices.columns
    vol_df.index =  ['Volatility (yearly percent)']
    return vol_df


vol_df = calculate_volatility(plot_df)



def yearly_return(prices: pd.DataFrame) -> pd.DataFrame:
    """
    The yearly return is the return if the asset where purchased on the
    first trading day and sold on the last trading day.
    :param prices: the asset prices for the time period.
    :return: a DataFrame containing the yearly returns for the asset.
    """
    year_periods = find_year_periods(prices)
    year_return_df = pd.DataFrame()
    for ix, period in year_periods.iterrows():
        start_ix = period['start_ix']
        end_ix = period['end_ix']
        start_val = prices[:][start_ix:start_ix + 1].values
        end_val = prices[:][end_ix:end_ix + 1].values
        r = (end_val/start_val) - 1
        r_percent = np.round(r * 100, 2)
        r_df = pd.DataFrame(r_percent)
        r_df.columns = prices.columns
        year_return_df = pd.concat([year_return_df, r_df])
    year_return_df.index = year_periods['year']
    return year_return_df


def yearly_drawdown(prices: pd.DataFrame) -> pd.DataFrame:
    year_periods = find_year_periods(prices)
    drawdown_df = pd.DataFrame()
    for ix, period in year_periods.iterrows():
        start_ix = period['start_ix']
        end_ix = period['end_ix']
        year_df = prices[:][start_ix:end_ix+1]
        year_drawdown = qs.stats.max_drawdown(year_df)
        year_drawdown_df = pd.DataFrame(year_drawdown).transpose()
        drawdown_df = pd.concat([drawdown_df, year_drawdown_df])
    drawdown_df = round(drawdown_df * 100, 2)
    drawdown_df.index = year_periods['year']
    return drawdown_df


def build_drawdown_return(prices: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    year_return_df = yearly_return(prices)
    drawdown_df = yearly_drawdown(prices)
    return_cols = list()
    for col in year_return_df.columns:
        ret_col = f'{col} return'
        return_cols.append(ret_col)
    year_return_df.columns = return_cols
    drawdown_cols = list()
    for col in drawdown_df.columns:
        dd_col = f'{col} drawdown'
        drawdown_cols.append(dd_col)
    drawdown_df.columns = drawdown_cols
    table_df = pd.concat([year_return_df, drawdown_df], axis=1)
    table_mean = pd.DataFrame(table_df.mean()).transpose()
    table_mean.index = ['average']
    return table_df, table_mean


table_df, table_mean = build_drawdown_return(plot_df)


d2010_start: datetime = datetime.fromisoformat('2010-01-04')

d2010_spy_close = spy_data.close_data(d2010_start, end_date)
d2010_moving_avg = spy_data.moving_avg(d2010_start, end_date)
plot_df = pd.concat([d2010_spy_close, d2010_moving_avg], axis=1)


plot_df.plot(grid=True, title=f'Moving average and SPY from {d2010_start.strftime("%m/%d/%Y")}', figsize=(10,6))


d2010_start: datetime = datetime.fromisoformat('2010-01-04')
d2010_spy_portfolio_df, assets_df = portfolio_return(holdings=holdings,
                                               risk_asset=spy_close,
                                               bond_asset=cash_trigger_bond_adjclose,
                                               spy_data=spy_data,
                                               start_date=d2010_start,
                                               end_date=end_date
                                               )

plot_df = build_plot_data(holdings=holdings, portfolio_df=d2010_spy_portfolio_df, spy_df=spy_close)
plot_df.plot(grid=True, title='SPY Portfolio from 2010 and SPY', figsize=(10,6))

vol_df = calculate_volatility(plot_df)

# print(tabulate(vol_df, headers=[*vol_df.columns], tablefmt='fancy_grid'))

table_df, table_mean = build_drawdown_return(plot_df)


d5_year_start = end_date - relativedelta(years=5)
d5_spy_close = spy_data.close_data(d5_year_start, end_date)
d5_moving_avg = spy_data.moving_avg(d5_year_start, end_date)
plot_df = pd.concat([d5_spy_close, d5_moving_avg], axis=1)

plot_df.plot(grid=True, title=f'Moving average and SPY from {d5_year_start.strftime("%m/%d/%Y")}', figsize=(10,6))

five_year_spy_portfolio_df, t = portfolio_return(holdings=holdings,
                                               risk_asset=spy_close,
                                               bond_asset=cash_trigger_bond_adjclose,
                                               spy_data=spy_data,
                                               start_date=d5_year_start,
                                               end_date=end_date
                                               )

plot_df = build_plot_data(holdings=holdings, portfolio_df=five_year_spy_portfolio_df, spy_df=spy_close)

vol_df = calculate_volatility(plot_df)
# print(tabulate(vol_df, headers=[*vol_df.columns], tablefmt='fancy_grid'))

table_df, table_mean = build_drawdown_return(plot_df)
# print(tabulate(table_df, headers=[*table_df.columns], tablefmt='fancy_grid'))

# print(tabulate(table_mean, headers=[*table_mean.columns], tablefmt='fancy_grid'))


d3_year_start = end_date - relativedelta(years=3)

d3_spy_close = spy_data.close_data(d3_year_start, end_date)
d3_moving_avg = spy_data.moving_avg(d3_year_start, end_date)
plot_df = pd.concat([d3_spy_close, d3_moving_avg], axis=1)

plot_df.plot(grid=True, title=f'Moving average and SPY from {d3_year_start.strftime("%m/%d/%Y")}', figsize=(10,6))


three_year_spy_portfolio_df, t = portfolio_return(holdings=holdings,
                                               risk_asset=spy_close,
                                               bond_asset=cash_trigger_bond_adjclose,
                                               spy_data=spy_data,
                                               start_date=d3_year_start,
                                               end_date=end_date
                                               )

plot_df = build_plot_data(holdings=holdings, portfolio_df=three_year_spy_portfolio_df, spy_df=spy_close)


d1_year_start = end_date - relativedelta(years=1)

d1_spy_close = spy_data.close_data(d1_year_start, end_date)
d1_moving_avg = spy_data.moving_avg(d1_year_start, end_date)
plot_df = pd.concat([d1_spy_close, d1_moving_avg], axis=1)

plot_df.plot(grid=True, title=f'Moving average and SPY from {d1_year_start.strftime("%m/%d/%Y")}', figsize=(10,6))


one_year_spy_portfolio_df, t = portfolio_return(holdings=holdings,
                                               risk_asset=spy_close,
                                               bond_asset=cash_trigger_bond_adjclose,
                                               spy_data=spy_data,
                                               start_date=d1_year_start,
                                               end_date=end_date
                                               )

plot_df = build_plot_data(holdings=holdings, portfolio_df=one_year_spy_portfolio_df, spy_df=spy_close)


market_etfs = ['SOXX', 'VV', 'VO', 'VB']

market_etf_file = "market_etf_adjclose"
market_etf_adjclose = get_market_data(file_name=market_etf_file,
                                      data_col='Adj Close',
                                      symbols=market_etfs,
                                      data_source=data_source,
                                      start_date=start_date,
                                      end_date=end_date)


def calc_asset_portfolio(holdings: float, prices_df: pd.DataFrame, weights: np.array) -> pd.DataFrame:
    """
     Given a set of asset prices and weights, return a DataFrame where each column contains
     a price series for the asset, starting with a weighted portfolio investment. For example,
     if the weights are [0.25, 0.25, 0.25, 0.25] and the initial holding is 100,000 the
     initial prices series will be the returns for each asset applied to 25,000. The
     portfolio will be rebalanced every year, so the weights will be approximately
     maintained.
    :param holdings: the initial value invested in the portfolio
    :param prices_df: a DataFrame containing a price series for the portfolio assets
    :param weights: The weights for the various assets
    :return: The prices series for the yearly rebalanced assets.
    """
    start_balance = holdings * weights
    portfolio_df = pd.DataFrame()
    year_periods = find_year_periods(prices_df)
    # columns=['start_ix', 'end_ix', 'year']
    for ix, period in year_periods.iterrows():
        start_ix = period['start_ix']
        end_ix = period['end_ix']
        year_prices_df = prices_df[:][start_ix:end_ix+1]
        year_return = return_df(year_prices_df)
        year_portfolio_df = pd.DataFrame([[], [], [], []]).transpose()
        year_portfolio_df.columns = prices_df.columns
        for jx, col in enumerate(year_prices_df.columns):
            col_prices_a = apply_return(start_balance[jx], year_return[col])
            year_portfolio_df[col] = pd.DataFrame(col_prices_a)
        year_total = year_portfolio_df.tail(1).values.sum()
        # rebalanced every year
        start_balance = year_total * weights
        portfolio_df = pd.concat([portfolio_df, year_portfolio_df])
    date_index = prices_df.index
    portfolio_df.index = date_index
    return portfolio_df



etf_weights = np.full(market_etf_adjclose.shape[1], 0.25)
portfolio_prices = calc_asset_portfolio(holdings=holdings, prices_df=market_etf_adjclose, weights=etf_weights)
portfolio_sum_s = portfolio_prices.sum(axis=1)
portfolio_sum_df = pd.DataFrame(portfolio_sum_s)
portfolio_sum_df.columns = ['portfolio']
plot_df = build_plot_data(holdings=holdings, portfolio_df=portfolio_sum_df, spy_df=spy_close)


four_etf_portfolio_df, t = portfolio_return(holdings=holdings,
                                               risk_asset=portfolio_sum_df,
                                               bond_asset=cash_trigger_bond_adjclose,
                                               spy_data=spy_data,
                                               start_date=start_date,
                                               end_date=end_date
                                               )

plot_df = build_plot_data(holdings=holdings, portfolio_df=four_etf_portfolio_df, spy_df=spy_close)

five_year_etf_portfolio_df, t = portfolio_return(holdings=holdings,
                                               risk_asset=portfolio_sum_df,
                                               bond_asset=cash_trigger_bond_adjclose,
                                               spy_data=spy_data,
                                               start_date=d5_year_start,
                                               end_date=end_date
                                               )

plot_df = build_plot_data(holdings=holdings, portfolio_df=five_year_etf_portfolio_df, spy_df=spy_close)


def percent_return_df(start_date: datetime, end_date: datetime, prices_df: pd.DataFrame) -> pd.DataFrame:
    def percent_return(time_series: pd.Series) -> pd.Series:
        return list(((time_series[i] / time_series[0]) - 1.0 for i in range(0, len(time_series))))


    date_index = prices_df.index
    start_ix = findDateIndex(date_index, start_date)
    end_ix = findDateIndex(date_index, end_date)
    period_df = prices_df[:][start_ix:end_ix+1]
    period_return_df = pd.DataFrame()
    for col in period_df.columns:
        return_series = percent_return(period_df[col])
        period_return_df[col] = return_series
    period_return_df.index = period_df.index
    return_percent_df = round(period_return_df * 100, 2)
    return return_percent_df


# We already have the bond set JNK, TLT, MUB or SHYin the DataFrame cash_trigger_bond_adjclose
# We already have SPY close prices. For consistency we'll use close prices, instead of
# adjusted close prices for IWM, MDY and QQQ.

rotation_etfs = ['IWM', 'MDY', 'QQQ']
rotation_etf_file = 'rotation_etf_close'
rotation_etf_close = get_market_data(file_name=rotation_etf_file,
                                      data_col='Close',
                                      symbols=rotation_etfs,
                                      data_source=data_source,
                                      start_date=start_date,
                                      end_date=end_date)

# SPY has already been downloaded. So add SPY to the rotation_etf_close set.
rotation_etf_close['SPY'] = spy_close
# Add SHY (the cash proxy bond) to the ETF rotation set.
rotation_etf_shy = rotation_etf_close.copy()
rotation_etf_shy[shy_ticker] = shy_adjclose


percent_ret = percent_return_df(start_date=start_date,
                                end_date=end_date,
                                prices_df=rotation_etf_close)
percent_ret.plot(grid=True, title='Percent return for the ETF Rotation set', figsize=(10,6))


etf_rotation_portfolio_df, t = portfolio_return(holdings=holdings,
                                               risk_asset=rotation_etf_shy,
                                               bond_asset=cash_trigger_bond_adjclose,
                                               spy_data=spy_data,
                                               start_date=start_date,
                                               end_date=end_date
                                               )

plot_df = build_plot_data(holdings=holdings,
                          portfolio_df=etf_rotation_portfolio_df,
                          spy_df=spy_close)

terminal_vals_s = plot_df[:].iloc[-1]
terminal_vals_df = pd.DataFrame(terminal_vals_s).transpose()


def get_asset_investments(risk_asset: pd.DataFrame,
                          bond_asset: pd.DataFrame,
                          spy_data: SpyData,
                          start_date: datetime,
                          end_date: datetime) -> pd.DataFrame:
    name_l: List = []
    risk_l: List = []
    date_l: List = []
    end_date_l: List = []
    month_periods = find_month_periods(start_date, end_date, risk_asset)
    for index, period in month_periods.iterrows():
        start_ix = period['start_ix']
        end_ix = period['end_ix']
        period_start_date: datetime = convert_date(period['start_date'])
        period_end_date: datetime = convert_date(period['end_date'])
        date_l.append(period_start_date)
        end_date_l.append(period_end_date)
        asset_name = ''
        if spy_data.risk_state(period_start_date) == RiskState.RISK_ON:
            asset_name: str = chooseAssetName(start_ix, end_ix, risk_asset)
            risk_on: bool = True
        else:  # RISK_OFF - bonds
            asset_name: str = chooseAssetName(start_ix, end_ix, bond_asset)
            risk_on: bool = False
        name_l.append(asset_name)
        risk_l.append(risk_on)
    asset_df = pd.DataFrame([name_l, date_l, end_date_l, risk_l]).transpose()
    asset_df.index = date_l
    asset_df.columns = ['asset', 'start_date', 'end_date', 'risk-on']
    asset_df = collapse_asset_df(asset_df=asset_df)
    return asset_df


def investment_return(holdings: float, investment_df: pd.DataFrame, prices_df: pd.DataFrame) -> pd.DataFrame:
    date_index = prices_df.index
    date_l: List = []
    prices_l: List = []
    balance = holdings
    first = True
    for ix, row in investment_df.iterrows():
        start_date = row['start_date']
        end_date = row['end_date']
        if first:
            prices_l.append(balance)
            date_l.append(start_date)
            first = False
        asset = row['asset']
        start_ix = findDateIndex(date_index, start_date)
        end_ix = findDateIndex(date_index, end_date)
        assert start_ix >= 0 and end_ix >= 0
        row_one = prices_df[asset][start_ix:start_ix+1]
        row_n = prices_df[asset][end_ix:end_ix+1]
        r = (row_n.values[0] / row_one.values[0]) -1
        balance = balance + balance * r
        prices_l.append(balance)
        date_l.append(end_date)
    portfolio_df = pd.DataFrame(prices_l)
    portfolio_df.index = date_l
    portfolio_df.columns = ['portfolio']
    return portfolio_df


asset_df = get_asset_investments(risk_asset=rotation_etf_shy,
                                 bond_asset=cash_trigger_bond_adjclose,
                                 spy_data=spy_data,
                                 start_date=start_date,
                                 end_date=end_date
                                 )

all_assets = pd.concat([rotation_etf_close, cash_trigger_bond_adjclose], axis=1)

new_portfolio_df = investment_return(holdings=holdings,
                                     investment_df=asset_df,
                                     prices_df=all_assets)

new_portfolio_adj_df, spy_period_adj = adjust_time_series(new_portfolio_df, spy_close)

plot_df = build_plot_data(holdings=holdings, portfolio_df=new_portfolio_adj_df, spy_df=spy_period_adj)




print("Hi there")
