"""Loader: read one ESA-ADB channel into a pandas Series."""
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "ESA-ADB" / "data"


def load_channel(mission: int, channel: int) -> pd.Series:
    """Return one channel as a Series: index = timestamps, values = sensor readings."""
    path = DATA / f"ESA-Mission{mission}" / "channels" / f"channel_{channel}.zip"
    df = pd.read_pickle(path)          # thaw the saved table
    return df[f"channel_{channel}"]    # table with 1 column -> just that column
