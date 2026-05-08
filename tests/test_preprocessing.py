import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..')
    )
)

import pandas as pd

from src.preprocessing import clean_data


def test_missing_values_removed():

    df = pd.DataFrame({
        'a': [1, None],
        'b': [2, 3]
    })

    cleaned = clean_data(df)

    assert cleaned.isnull().sum().sum() == 0