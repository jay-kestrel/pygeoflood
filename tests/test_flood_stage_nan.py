"""NaN-discharge exclusion in get_flood_stage (review art_SjgX1Jzb F8).

A SRC row with NaN Discharge (missing COMID or roughness) poisons
np.interp: every query at or after the NaN x-coordinate silently returns
NaN, so affected segments vanish from the inundation map. The fix warns
and excludes the NaN rows before interpolation.
"""

import numpy as np
import pandas as pd
import pytest

from pygeoflood.tools import get_flood_stage


def src_frame():
    """Three hydroids: a fully valid curve, a partially NaN curve, and a
    hydroid whose every row is NaN (roughness join missed entirely)."""
    stages = np.round(np.arange(0.0, 2.01, 0.1), 4)
    rows = []
    for stage in stages:
        rows.append(
            {
                "HYDROID": 1,
                "Stage_m": stage,
                "COMID": 100.0,
                "Discharge_cms": 5.0 * stage,
            }
        )
    for stage in stages:
        discharge = 5.0 * stage if stage < 1.0 else np.nan
        rows.append(
            {
                "HYDROID": 2,
                "Stage_m": stage,
                "COMID": 100.0,
                "Discharge_cms": discharge,
            }
        )
    for stage in stages:
        rows.append(
            {
                "HYDROID": 3,
                "Stage_m": stage,
                "COMID": 200.0,
                "Discharge_cms": np.nan,
            }
        )
    return pd.DataFrame(rows)


def test_nan_discharge_rows_are_excluded_and_warned():
    with pytest.warns(UserWarning, match="NaN Discharge"):
        df = get_flood_stage(src_frame(), None, 7.5)
    stages = df.set_index("HYDROID")["Stage_m"]
    # hydroid 1: fully valid curve — 7.5 cms sits at stage 1.5
    assert float(stages.loc[1]) == pytest.approx(1.5)
    # hydroid 2: valid rows only reach 4.5 cms -> above the curve -> the
    # nodata right-cliff convention, NOT a NaN-poisoned stage
    assert float(stages.loc[2]) == -32768.0
    # hydroid 3: no valid rows remain -> undefined stage (NaN row kept)
    assert bool(np.isnan(stages.loc[3]))
    assert len(df) == 3  # row set unchanged: exclusion, not deletion


def test_nan_rows_cannot_poison_other_hydroids():
    """The review's core scenario: hydroid 1's stage stays finite even
    though other hydroids' lookups contain NaN rows."""
    with pytest.warns(UserWarning, match="NaN Discharge"):
        df = get_flood_stage(src_frame(), None, 7.5)
    stages = df.set_index("HYDROID")["Stage_m"]
    assert stages.notna().loc[1]
    assert float(stages.loc[1]) == pytest.approx(1.5)
