import numpy as np

from .base import MultivariateOutlierGenerator


class MultivariateFreqOutlierGenerator(MultivariateOutlierGenerator):
    def __init__(self, timestamps=None,
                       phase_shift=np.random.uniform(0, 2*np.pi),
                       amplitude=np.random.uniform(0, 3),
                       freq = np.random.uniform(0, 0.5)):
        self.timestamps = timestamps or []
        self.phase_shift = phase_shift
        self.amplitude = amplitude
        self.freq = freq

    def add_outliers(self, timeseries):
        additional_values = np.zeros(timeseries.size)
        for start, end in self.timestamps:
            # difference = np.diff(timeseries[start-1:end]) if start > 0 \
            #              else np.insert(np.diff(timeseries[start:end]), 0, 0)

            x = np.array(list(range(start, end)))

            additional_values[list(range(start, end))] += self.amplitude * np.cos(2*np.pi*self.freq*x+self.phase_shift)

        return additional_values