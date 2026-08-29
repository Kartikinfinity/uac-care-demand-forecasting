import numpy as np
import pandas as pd

class NaiveBaseline:
    """Predicts the last observed value (t-1)."""
    def __init__(self):
        self.last_value_ = None
        
    def fit(self, y):
        # We only need the very last valid value.
        # If there's an index, just take the raw values
        y_vals = np.asarray(y)
        # Handle trailing NaNs if any, by getting the last non-NaN
        valid = y_vals[~np.isnan(y_vals)]
        if len(valid) == 0:
            self.last_value_ = np.nan
        else:
            self.last_value_ = valid[-1]
        return self
        
    def predict(self, horizons):
        """
        horizons: integer or list of integers (e.g. 1, 7, 14)
        Returns the naive prediction for each horizon.
        """
        if isinstance(horizons, int):
            return self.last_value_
        return np.array([self.last_value_] * len(horizons))


class SeasonalNaiveBaseline:
    """Predicts the value from m periods ago (e.g., t-5 for m=5)."""
    def __init__(self, m=5):
        self.m = m
        self.history_ = None
        
    def fit(self, y):
        y_vals = np.asarray(y)
        # Store at least m valid observations, but dealing with NaNs requires care.
        # Since this is a baseline evaluated identically, we store the full series.
        self.history_ = y_vals
        return self
        
    def predict(self, horizons):
        """
        If h = 1, predicts history[-m] (if m=5, history[-5])
        If h = 2, predicts history[-m+1]
        If h = m, predicts history[-1]
        If h = m+1, predicts history[-m] again
        """
        res = []
        is_scalar = isinstance(horizons, int)
        h_list = [horizons] if is_scalar else horizons
        
        for h in h_list:
            # We want the offset looking backward from the end.
            # When h=1, we look exactly m steps back: idx = -m
            # When h=m, we look 1 step back: idx = -1
            # In general, offset from the end is -(m - (h - 1) % m)
            offset = -(self.m - (h - 1) % self.m)
            
            if len(self.history_) < abs(offset):
                res.append(np.nan)
            else:
                res.append(self.history_[offset])
                
        if is_scalar:
            return res[0]
        return np.array(res)


class MovingAverageBaseline:
    """Predicts the mean of the most recent w periods."""
    def __init__(self, w=7):
        self.w = w
        self.mean_val_ = None
        
    def fit(self, y):
        y_vals = np.asarray(y)
        # Get the last w periods
        window = y_vals[-self.w:]
        # Calculate mean ignoring NaNs
        self.mean_val_ = np.nanmean(window) if len(window) > 0 else np.nan
        return self
        
    def predict(self, horizons):
        if isinstance(horizons, int):
            return self.mean_val_
        return np.array([self.mean_val_] * len(horizons))

