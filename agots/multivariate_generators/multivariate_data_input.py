import numpy as np
import pandas as pd

from agots.multivariate_generators.multivariate_extreme_outlier_generator import MultivariateExtremeOutlierGenerator
from agots.multivariate_generators.multivariate_shift_outlier_generator import MultivariateShiftOutlierGenerator
from agots.multivariate_generators.multivariate_trend_outlier_generator import MultivariateTrendOutlierGenerator
from agots.multivariate_generators.multivariate_variance_outlier_generator import MultivariateVarianceOutlierGenerator

from agots.generators.extreme_outlier_generator import ExtremeOutlierGenerator
from agots.generators.shift_outlier_generator import ShiftOutlierGenerator
from agots.generators.trend_outlier_generator import TrendOutlierGenerator
from agots.generators.variance_outlier_generator import VarianceOutlierGenerator

INITIAL_VALUE_MIN = 0
INITIAL_VALUE_MAX = 1

class MultivariateDataInput:
    def __init__(self, df):
        """Create multivariate time series using outlier generators
        :param stream_length: number of values in each time series
        :param n: number of time series at all
        :param k: number of time series that should correlate. If all should correlate with the first
        one, set k=n.
        :param shift_config: dictionary from index of the time series to how much it should be displaced in time (>=0)
        """

        # n, k, stream_length

        self.data = df
        self.N = len(df.columns)
        self.outlier_data = pd.DataFrame()

    def add_outliers(self, config):
        """Adds outliers based on the given configuration to the base line

         :param config: Configuration file for the outlier addition e.g.
         {'extreme': [{'n': 0, 'timestamps': [(3,)]}],
          'shift':   [{'n': 3, 'timestamps': [(4,10)]}]}
          would add an extreme outlier to time series 0 at timestamp 3 and a base shift
          to time series 3 between timestamps 4 and 10
         :return:
         """
        OUTLIER_GENERATORS = {'extreme': MultivariateExtremeOutlierGenerator,
                              'shift': MultivariateShiftOutlierGenerator,
                              'trend': MultivariateTrendOutlierGenerator,
                              'variance': MultivariateVarianceOutlierGenerator}

        generator_keys = []

        # Validate the input
        for outlier_key, outlier_generator_config in config.items():
            assert outlier_key in OUTLIER_GENERATORS, 'outlier_key must be one of {} but was'.format(OUTLIER_GENERATORS,
                                                                                                     outlier_key)
            generator_keys.append(outlier_key)
            for outlier_timeseries_config in outlier_generator_config:
                n, timestamps = outlier_timeseries_config['n'], outlier_timeseries_config['timestamps']
                assert n in range(self.N), 'n must be between 0 and {} but was {}'.format(self.N - 1, n)
                for timestamp in list(sum(timestamps, ())):
                    assert timestamp in range(
                        len(self.data)), 'timestamp must be between 0 and {} but was {}'\
                        .format(len(self.data) - 1, timestamp)

        df = self.data.copy()

        if self.data.shape == (0, 0):
            raise Exception('Shape of Input DataFrame is (0, 0)!')
        for generator_key in generator_keys:
            for outlier_timeseries_config in config[generator_key]:
                n, timestamps = outlier_timeseries_config['n'], outlier_timeseries_config['timestamps']
                generator_args = dict([(k, v) for k, v in outlier_timeseries_config.items() if k not in ['n', 'timestamps']])
                generator = OUTLIER_GENERATORS[generator_key](timestamps=timestamps, **generator_args)
                df[df.columns[n]] += generator.add_outliers(self.data[self.data.columns[n]])

        assert not df.isnull().values.any(), 'There is at least one NaN in the generated DataFrame'
        self.outlier_data = df
        return df