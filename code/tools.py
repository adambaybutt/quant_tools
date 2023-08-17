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
    def calcCumulativeReturn(cls, returns: np.array,
        annualized: bool=False,
        periods_in_year: int=None) -> float: 
        """ Calculate the cumulative return of a vector of simple returns.

        Args:
            returns (np.array): vector of a simple returns at any frequency.
            annualized (bool): whether to annualize the statistic.
            periods_in_year (int): how many periods of the given frequency are in a year.

        Returns:
            (float): scalar cumulative return, annualized if requested.
        """
        # Run checks on input args
        cls.period_check(annualized, periods_in_year)
        
        # Calc
        cumulative_return = np.prod(1 + returns) - 1
        
        # Annualized
        if annualized:
            num_years_of_returns = len(returns) / periods_in_year
            cum_ret_annl = cumulative_return / num_years_of_returns
            if cum_ret_annl < -1:
                return -1
            else:
                return cum_ret_annl
        else:
            return cumulative_return
    
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
            (float): scalar annualized Sharpe ratio.
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

    @staticmethod
    def calcTSAvgTurnover(df: pd.DataFrame) -> float:
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

    @staticmethod
    def calcTStatReturns(returns: np.array, null: float = 0) -> float:
        """
        Calculate the t-statistic for a vector of simple returns using the formula:
        t = sqrt(N) * (mean_return - null) / std_dev_return

        Args:
            returns (np.array): vector of a simple returns at any frequency.
            null (float): null hypothesis value.

        Returns:
            (float): scalar t-statistic.
        """
        mean_return = QuantTools.calcTSAvgReturn(returns)
        std_dev_return = QuantTools.calcSD(returns)

        if std_dev_return == 0:
            raise ValueError("Standard deviation is zero, cannot calculate t-statistic.")

        sample_size = len(returns)

        return (np.sqrt(sample_size) * (mean_return - null) / std_dev_return)

    @staticmethod
    def formPortfolioWeightsByQuantile(
        df: pd.DataFrame, num_qntls_prtls: int, mcap_weighted: bool=False
        ) -> pd.DataFrame:
        """
        Creates a new column "prtfl_wght_hml" representing portfolio allocation percentages
        within a date, long the top quantile and short the bottom quantile, and optionally 
        weighted by market capitalization ("mcap").

        Args:
            df (pd.DataFrame): DataFrame containing "date", "asset", "yhats", and optionally "mcap".
            num_qntls_prtls (int): Number of quantiles to form (must be greater than 1).
            mcap_weighted (bool): If True, weight "prtfl_wght_hml" by market capitalization.

        Returns:
            pd.DataFrame: Modified DataFrame with new portfolio weight column(s).
        """
        # Validate input
        required_columns = ['date', 'asset', 'yhats'] + (['mcap'] if mcap_weighted else [])
        if not (set(required_columns) <= set(df.columns)) or num_qntls_prtls < 2:
            raise ValueError(f"Input DataFrame must contain columns {required_columns}, and 'num_qntls_prtls' must be greater than 1")
    
        # Run check on input args
        if not isinstance(num_qntls_prtls, int) or num_qntls_prtls < 2:
            raise ValueError("Input 'num_qntls_prtls' must be an integer greater than 1")
        
        # Randomly sort the DataFrame and then by yhats to randomly assign ties
        df = df.sample(frac=1).sort_values(by=['date', 'yhats'], ignore_index=True)

        # Calculate quantiles based on a noisy yhat
        df['quantile'] = df.groupby('date')['yhats'].transform(
            lambda x: 1+pd.qcut(x + np.random.uniform(-1e-10, 1e-10, size=len(x)), num_qntls_prtls, labels=False))

        # Assign long or short sign
        df['prtfl_wght_hml'] = np.where(df['quantile'] == 1, -1, np.where(df['quantile'] == num_qntls_prtls, 1, 0))

        # Update prtfl_wght_hml to equal weighted or mcap weighted 
        if mcap_weighted:
            # Form high minus low portfolio weight
            df['prtfl_wght_hml'] *= df['mcap']
            df['prtfl_wght_hml'] = df.groupby(['date', 'quantile'], group_keys=False)['prtfl_wght_hml'].apply(lambda x: x / x.abs().sum())
            df.loc[df.prtfl_wght_hml.isnull(), 'prtfl_wght_hml'] = 0

            # Also form portfolio weight within each quantile
            df['prtfl_wght_mcap'] = df.groupby(['date', 'quantile'], group_keys=False)['mcap'].apply(lambda x: x / x.sum())

            # Check that the prtfl_wght column sums to 1 within each date-quantile
            assert all(np.isclose(df.groupby(['date', 'quantile'])['prtfl_wght_mcap'].sum(), 1)), \
                "prtfl_wght_mcap column sums do not equal 1 for all date-quantiles"
        else:
            df['prtfl_wght_hml'] = df.groupby(
                ['date', 'prtfl_wght_hml'])['prtfl_wght_hml'].transform(lambda x: x / x.count())

        # Check that the prtfl_wght_hml column sums to 1 within each date
        assert len(np.unique(df.date.values)) == np.sum(np.isclose(df.groupby('date')['prtfl_wght_hml'].sum(), 0)), \
            "prtfl_wght_hml column sums do not equal 0 for all dates"

        # Sort the DataFrame by date and asset
        df = df.sort_values(by=['date', 'asset'], ignore_index=True)

        # Return DataFrame with relevant columns
        return df

    @staticmethod
    def calcR2Pred(ys: np.array, yhats: np.array) -> float:
        """
        Calculates the R-squared prediction value.

        :param ys: numpy array of actual target values
        :param yhats: numpy array of predicted target values

        :return: R-squared prediction value
        """
        residual_variance = np.mean(np.square(ys - yhats))
        total_variance = np.mean(np.square(ys))
        
        return 1 - residual_variance / total_variance
