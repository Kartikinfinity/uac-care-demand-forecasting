import numpy as np
import pytest
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.models.baselines import NaiveBaseline, SeasonalNaiveBaseline, MovingAverageBaseline

def test_naive_baseline():
    y = [10, 20, np.nan, 40]
    model = NaiveBaseline()
    model.fit(y)
    
    assert model.predict(1) == 40
    np.testing.assert_array_equal(model.predict([1, 7]), [40, 40])

def test_seasonal_naive_baseline():
    # m = 5
    y = [1, 2, 3, 4, 5, 11, 12, 13, 14, 15]
    model = SeasonalNaiveBaseline(m=5)
    model.fit(y)
    
    # h=1 -> looks back m=5 -> 11
    assert model.predict(1) == 11
    # h=2 -> looks back m-1=4 -> 12
    assert model.predict(2) == 12
    # h=5 -> looks back 1 -> 15
    assert model.predict(5) == 15
    # h=6 -> looks back 5 -> 11
    assert model.predict(6) == 11
    
    preds = model.predict([1, 7])
    np.testing.assert_array_equal(preds, [11, 12])

def test_moving_average_baseline():
    y = [10, 20, 30, 40, 50]
    model = MovingAverageBaseline(w=3)
    model.fit(y)
    
    # Mean of last 3: 30, 40, 50 -> 40
    assert model.predict(1) == 40
    np.testing.assert_array_equal(model.predict([1, 7]), [40, 40])
