import numpy as np

from .base import MultivariateOutlierGenerator


class MultivariateExtremeOutlierGenerator(MultivariateOutlierGenerator):
    def __init__(self, timestamps=None, factor=8, random=True, back=10, front=10):
        self.timestamps = [] if timestamps is None else list(sum(timestamps, ()))
        self.factor = factor
        self.back = back
        self.front = front

        if random:
            self.updown = np.random.choice([-1, 1])
        else:
            self.updown = 1 #following factor

    def get_value(self, current_timestamp, timeseries):
        if current_timestamp in self.timestamps:
            local_std = timeseries.iloc[max(0, current_timestamp - self.back):current_timestamp + self.front].std()
            return self.updown * self.factor * local_std
        else:
            return 0

    def add_outliers(self, timeseries):
        additional_values = []
        for timestamp_index in range(len(timeseries)):
            additional_values.append(self.get_value(timestamp_index, timeseries))
        return additional_values
