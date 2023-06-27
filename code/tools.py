import pandas as pd
import numpy as np

def input_checks(method):
    def wrapper(cls, returns, *args, **kwargs):
        if not isinstance(returns, (np.ndarray, pd.Series)):
            raise TypeError("Input 'returns' must be either a NumPy array or a Pandas Series")
        if np.isnan(returns).any():
            raise ValueError("Input 'returns' contains NaN values")
        if returns.size == 0:
            raise ValueError("Input 'returns' must not be empty")
        return method(cls, returns, *args, **kwargs)
    return wrapper

class QuantTools:
    """ Class of helper functions for quantitative finance analyses. """

    @staticmethod
    def period_check(annualized, periods_in_year):
        if annualized and periods_in_year is None:
            raise ValueError("Input 'periods_in_year' must be provided if 'annualized' is True")
        if periods_in_year is not None and not (isinstance(periods_in_year, int) and periods_in_year > 0):
            raise ValueError("Input 'periods_in_year' must be a positive integer")

    @classmethod
    @input_checks
    def calcGeomAvg(cls, returns: np.array,
        annualized: bool=False,
        periods_in_year: int=None) -> float: 
        """ Calculate the geometric average of a vector of simple returns.

        Args:
            returns (np.array): vector of a simple returns at any frequency.
            annualized (bool): whether to annualize the statistic.
            periods_in_year (int): how many periods of the given frequency are in a year.

        Returns:
            (float): scalar geometric average.
        """
        # Run checks on input args
        cls.period_check(annualized, periods_in_year)
        
        # Calc
        geom_avg_at_given_freq = np.prod(1 + returns) ** (1 / np.size(returns)) - 1
        return (geom_avg_at_given_freq + 1) ** periods_in_year - 1 if annualized else geom_avg_at_given_freq
    
    @classmethod
    @input_checks
    def calcTSAvgReturn(cls, returns: np.array,
        annualized: bool=False,
        periods_in_year: int=None) -> float:
        """ Calculate the time series mean return of a vector of simple returns with option to annualize.

        Args:
            returns (np.array): vector of a simple returns at any frequency.
            annualized (bool): whether to annualize the statistic.
            periods_in_year (int): how many periods of the given frequency are in a year.

        Returns:
            (float): scalar time series mean return.
        """
        # run checks on input args
        cls.period_check(annualized, periods_in_year)
        
        # calc
        mean_ret_at_given_freq = np.mean(returns)
        if annualized == False:
            return mean_ret_at_given_freq
        else:
            mean_ret = periods_in_year*mean_ret_at_given_freq
            return max(mean_ret, -1.0)
    
    @classmethod
    @input_checks
    def calcSD(cls, returns: np.array,
        annualized: bool=False,
        periods_in_year: int=None) -> float: 
        """ Calculate the standard deviation of a vector of simple returns with option to annualize.

        Args:
            returns (np.array): vector of a simple returns at any frequency.
            annualized (bool): whether to annualize the statistic.
            periods_in_year (int): how many periods of the given frequency are in a year.

        Returns:
            (float): scalar standard deviation.
        """
        # run checks on input args
        cls.period_check(annualized, periods_in_year)
        
        # calc
        sd_at_given_freq = np.std(returns)
        return np.sqrt(periods_in_year) * sd_at_given_freq if annualized else sd_at_given_freq

    @classmethod
    @input_checks
    def calcSharpe(cls, returns: np.array,
        periods_in_year: int,
        risk_free_returns: np.array=None) -> float:
        """ Calculate the annual Sharpe Ratio of a vector of simple returns. 

        Args:
            returns (np.array): vector of a simple returns at any frequency.
            periods_in_year (int): how many periods of the given frequency are in a year.
            risk_free_returns (np.array): vector of simple returns of the risk free rate.

        Returns:
            (float): scalar standard deviation.
        """
        # run checks on input args and adjust returns if risk free given
        cls.period_check(False, periods_in_year)
        if risk_free_returns is not None:
            if not isinstance(risk_free_returns, np.ndarray):
                raise TypeError("Input 'risk_free_returns' must be a NumPy array")
            if np.isnan(risk_free_returns).any():
                raise ValueError("Input 'risk_free_returns' contains NaN values")
            if risk_free_returns.size != returns.size:
                raise ValueError("'returns' and 'risk_free_returns' must be of the same size")
            
            returns = returns - risk_free_returns
        
        # calc
        return (cls.calcTSAvgReturn(returns, annualized=True, periods_in_year=periods_in_year) /
                cls.calcSD(returns, annualized=True, periods_in_year=periods_in_year))
    
    @classmethod
    @input_checks
    def calcMaxDrawdown(cls, returns: np.array) -> float:
        ''' Calculate maximum drawdown for a vector of returns of any frequency.

        Args:
            returns (np.array): vector of simple returns.

        Returns:
            max_drawdown (float): maximum drawdown in simple return units over this period.
        '''
        # calculate the cumulative return as a new vector of the same length
        cumulative_ret = np.cumprod(1 + returns)

        # for every period, calc the historic maximum value of the portfolio 
        roll_max = np.maximum.accumulate(cumulative_ret)

        # calc max drawdown, in simple return, as current return divided by the historic max value minus 1
        drawdowns = cumulative_ret / roll_max - 1
        max_drawdown = np.min(drawdowns)
        
        return max_drawdown

    @classmethod
    def calcTSAvgTurnover(cls, df: pd.DataFrame) -> float:
        """
        This function takes a pandas DataFrame with columns "date", "asset", and "position" and
        calculates the average turnover, which is defined as the time series average of the percentage
        of assets each date that do not have the same position (-1, 0, 1) as the previous date for that asset.
        
        Args:
            df (pd.DataFrame): A pandas DataFrame containing columns "date", "asset", and "position".
        
        Returns:
            float: The average turnover.
        """
        # Check dataframe has required columns
        required_columns = ['date', 'asset', 'position']
        if not all(column in df.columns for column in required_columns):
            raise ValueError(f"Input DataFrame must contain columns {required_columns}")
        
        # Sort the DataFrame by date and asset
        df = df.sort_values(by=['date', 'asset'])

        # Shift the position column to get the previous position for each asset
        df['prev_position'] = df.groupby('asset')['position'].shift(1)

        # Calculate the percentage of assets each date that do not have the same position as the previous date
        df['position_changed'] = np.where(df['position'] != df['prev_position'], 1, 0)
        turnover_pct = df.groupby('date')['position_changed'].mean()

        # Calculate the time series average of the percentage of assets with changed positions
        average_turnover = turnover_pct.mean()

        return average_turnover

    @classmethod
    def formPortfolioPositionsQuantileLongShort(cls, df: pd.DataFrame, quantile: int) -> pd.DataFrame:
        """
        This function takes a pandas DataFrame with columns "date", "asset", and "yhats" and performs
        the specified steps to form a new column "position" containing portfolio allocation percentages
        that sum to 0 within date where we long the top quantile and short the bottom quantile.
        The function then returns the modified DataFrame with the four columns.
        
        Args:
            df (pd.DataFrame): A pandas DataFrame containing columns "date", "asset", and "yhats".
        
        Returns:
            pd.DataFrame: The modified DataFrame with an additional "position" column.
        """
        # Run checks on input data
        required_columns = ['date', 'asset', 'yhats']
        if not all(column in df.columns for column in required_columns):
            raise ValueError(f"Input DataFrame must contain columns {required_columns}")

        if not isinstance(quantile, int) or quantile < 2:
            raise ValueError("Input 'quantile' must be an integer greater than 1")
        
        # Randomly sort the DataFrame and then by yhats to randomly assign ties
        df = df.sample(frac=1).reset_index(drop=True)
        df = df.sort_values(by=['date', 'yhats'])

        # Add a small random noise to yhats to handle potential duplicated values
        df['yhats_noisy'] = df['yhats'] + np.random.uniform(-1e-10, 1e-10, size=len(df))

        # Calculate quantiles
        quantiles = df.groupby('date')['yhats_noisy'].transform(lambda x: pd.qcut(x, quantile, labels=False))

        # Assign positions
        df['position'] = np.where(quantiles == 0, -1, np.where(quantiles == quantile-1, 1, 0))

        # Divide the positions column by how many assets are in each date
        df['position'] = df.groupby(['date', 'position'])['position'].transform(lambda x: x / x.count())

        # Check that the position column sums to 1 within each date
        position_sums = df.groupby('date')['position'].sum()
        assert (all(np.isclose(position_sums, 0)), 
                f"Position column sums do not equal 1 for all dates: {position_sums[np.isclose(position_sums, 0)]}")

        # Drop the noisy yhats column
        df = df.drop(columns=['yhats_noisy'])

        # Sort the DataFrame by date and asset
        df = df.sort_values(by=['date', 'asset'])

        return df
    