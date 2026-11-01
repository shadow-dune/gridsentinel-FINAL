"""Day-ahead spot price predictor trained on REAL history (PRICE_HISTORY_PATH: timestamp, spot_eur_per_kwh).

History comes from Energy-Charts (data/price_history.py); timestamps are timezone-aware UTC.
Role in control: data/market.py uses it only for horizon steps whose day-ahead auction result is not yet published
(the next day's prices appear ~12:45 CET). The full-horizon prediction is also shown next to the published prices.

German prices swing strongly from day to day with wind and solar output, so time-of-day alone is a poor predictor.
Every candidate therefore uses recent published prices (lags of >= 1 day, which are always published for a 12-hour
horizon). Candidates are scored walk-forward on the last 7 days (2 if history is short); the lowest MAE wins and
the result says which. On real DE-LU data (Sep-Nov 2026) "same_time_yesterday" won: day-to-day swings come from
wind/solar output, which none of these features see. Adding a renewable-generation forecast is the obvious next step.
No invented fallback: if history is missing, too short, too old or lacks the lags, it raises PredictionUnavailable.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
STEPS_PER_DAY = 96
DAY = pd.Timedelta(days=1)
WEEK = pd.Timedelta(days=7)


class PredictionUnavailable(Exception):
    """Raised when historical price data is missing, too short, stale, or lacks the lags a prediction needs."""


@dataclass(frozen=True)
class PriceForecast:
    prices: List[float]            # spot EUR/kWh, aligned 1:1 with the requested timestamps
    model: str                     # candidate that won the held-out backtest
    mae: Dict[str, float]          # held-out MAE (EUR/kWh) per candidate
    trained_rows: int
    history_end: datetime
    lower_bound: List[float] = field(default_factory=list)  # 15th-percentile empirical band (EUR/kWh)
    upper_bound: List[float] = field(default_factory=list)  # 85th-percentile empirical band (EUR/kWh)


def _lags(series: pd.Series, ts: pd.DatetimeIndex) -> pd.DataFrame:
    """Lag features for target times `ts` (UTC) from a 15-min UTC series. NaN where history is missing."""
    level = series.rolling("1D").mean()  # mean of the 24 h ending at each timestamp
    return pd.DataFrame({
        "lag_1d": series.reindex(ts - DAY).to_numpy(),
        "lag_7d": series.reindex(ts - WEEK).to_numpy(),
        "level_1d": level.reindex(ts - DAY).to_numpy(),
    }, index=ts)


def _time_features(ts_local: pd.DatetimeIndex) -> pd.DataFrame:
    mod = (ts_local.hour * 60 + ts_local.minute).to_numpy(dtype=float)
    dow = ts_local.dayofweek.to_numpy(dtype=float)
    feats = {"dow": dow, "weekend": (dow >= 5).astype(float)}
    for h in (1, 2, 3):
        ang = 2.0 * np.pi * h * mod / 1440.0
        feats[f"sin{h}"], feats[f"cos{h}"] = np.sin(ang), np.cos(ang)
    return pd.DataFrame(feats, index=ts_local)


class _Yesterday:
    """Same 15-min slot yesterday (published). The baseline every other candidate must beat."""

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "_Yesterday":
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X["lag_1d"].to_numpy(dtype=float)


class _ProfileLevel:
    """Typical intraday shape (median by weekday/weekend and slot, minus its daily mean) + yesterday's level."""

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "_ProfileLevel":
        df = pd.DataFrame({"w": X["weekend"].to_numpy(), "slot": X["slot"].to_numpy(), "y": y - X["level_1d"].to_numpy()})
        self.table = df.groupby(["w", "slot"])["y"].median().to_dict()
        self.fallback = float(np.median(df["y"]))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        shape = [self.table.get((w, s), self.fallback) for w, s in zip(X["weekend"].to_numpy(), X["slot"].to_numpy())]
        return X["level_1d"].to_numpy(dtype=float) + np.array(shape, dtype=float)


class _Forest:
    """Random Forest on time-of-day harmonics, weekday and the published lags."""

    COLS = ["sin1", "cos1", "sin2", "cos2", "sin3", "cos3", "dow", "weekend", "lag_1d", "lag_7d", "level_1d"]

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "_Forest":
        from sklearn.ensemble import RandomForestRegressor

        self.m = RandomForestRegressor(n_estimators=150, max_depth=10, min_samples_leaf=4, random_state=42,
                                       n_jobs=-1).fit(X[self.COLS], y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.m.predict(X[self.COLS]), dtype=float)


class PredictiveModel:
    def __init__(self, csv_path: str, tz: str, min_days: int = 9, max_history_age_days: float = 7.0) -> None:
        self.tz = tz
        self.csv_path = csv_path
        self.min_rows = min_days * STEPS_PER_DAY  # 7 days for the weekly lag + 2 held-out days
        self.max_age = timedelta(days=max_history_age_days)
        self._series: Optional[pd.Series] = None
        self._best: Optional[str] = None
        self._fitted: Optional[object] = None
        self._mae: Dict[str, float] = {}
        self._q15_resid: float = 0.0
        self._q85_resid: float = 0.0
        self._trained_mtime: Optional[float] = None

    def _load(self) -> pd.Series:
        if not os.path.exists(self.csv_path):
            raise PredictionUnavailable(f"{self.csv_path} not found; run fetch_history.py to download real history")
        try:
            df = pd.read_csv(self.csv_path, usecols=["timestamp", "spot_eur_per_kwh"])
            raw_ts = df["timestamp"].astype(str)
            if not raw_ts.str.contains(r"(?:[+-]\d{2}:\d{2}|Z)$").all():
                raise ValueError("timestamps must carry a UTC offset (naive timestamps are ambiguous)")
            df["timestamp"] = pd.to_datetime(raw_ts, utc=True, errors="coerce")
            df["spot_eur_per_kwh"] = pd.to_numeric(df["spot_eur_per_kwh"], errors="coerce")
        except (ValueError, KeyError) as exc:
            raise PredictionUnavailable(f"history file unreadable: {exc}") from exc
        df = df.dropna().drop_duplicates("timestamp").sort_values("timestamp")
        df = df[np.isfinite(df["spot_eur_per_kwh"])]  # negative spot prices are real; keep them
        if len(df) < self.min_rows:
            raise PredictionUnavailable(f"only {len(df)} rows of history; need at least {self.min_rows}")
        s = df.set_index("timestamp")["spot_eur_per_kwh"]
        # hourly stretches (older data / other zones) are forward-filled onto the 15-min grid
        return s.resample("15min").ffill(limit=3).dropna()

    def _design(self, series: pd.Series, ts: pd.DatetimeIndex) -> pd.DataFrame:
        local = ts.tz_convert(self.tz)
        tf = _time_features(local)
        tf.index = ts
        X = _lags(series, ts).join(tf)
        X["slot"] = ((local.hour * 60 + local.minute) // 15).to_numpy()
        return X

    def _candidates(self) -> Dict[str, type]:
        c: Dict[str, type] = {"same_time_yesterday": _Yesterday, "profile_plus_level": _ProfileLevel}
        try:
            import sklearn  # noqa: F401
            c["random_forest"] = _Forest
        except ImportError:
            logger.info("scikit-learn not installed; random_forest candidate skipped")
        return c

    def train(self) -> Dict[str, float]:
        """Walk-forward check: fit on all but the last 7 (or 2) days, score on them, then refit the winner on everything."""
        mtime = os.path.getmtime(self.csv_path) if os.path.exists(self.csv_path) else None
        series = self._load()
        X = self._design(series, series.index).dropna()
        y = series.reindex(X.index).to_numpy(dtype=float)
        if len(X) < 3 * STEPS_PER_DAY:
            raise PredictionUnavailable("not enough history with 7-day lags to train")
        holdout_days = 7 if len(X) >= 21 * STEPS_PER_DAY else 2
        cut = len(X) - holdout_days * STEPS_PER_DAY
        candidates = self._candidates()
        mae: Dict[str, float] = {}
        resid: Dict[str, np.ndarray] = {}
        for name, cls in candidates.items():
            pred = cls().fit(X.iloc[:cut], y[:cut]).predict(X.iloc[cut:])
            resid[name] = y[cut:] - pred
            mae[name] = float(np.mean(np.abs(resid[name])))
        best = min(candidates, key=lambda n: mae[n])
        self._fitted = candidates[best]().fit(X, y)
        self._q15_resid = float(np.percentile(resid[best], 15))
        self._q85_resid = float(np.percentile(resid[best], 85))
        self._series, self._best = series, best
        self._mae = {k: round(v, 4) for k, v in mae.items()}
        self._trained_mtime = mtime
        logger.info("Price model %s trained on %d real rows; held-out MAE %s", best, len(X), self._mae)
        return self._mae

    def predict(self, timestamps: Sequence[datetime]) -> PriceForecast:
        """Predict spot prices for the ACTUAL horizon timestamps."""
        mtime = os.path.getmtime(self.csv_path) if os.path.exists(self.csv_path) else None
        if self._fitted is None or self._series is None or mtime != self._trained_mtime:
            self.train()
        assert self._series is not None and self._fitted is not None and self._best is not None

        ts = pd.DatetimeIndex(pd.to_datetime(list(timestamps), utc=True))
        end = self._series.index.max().to_pydatetime()
        if ts.min().to_pydatetime() - end > self.max_age:
            raise PredictionUnavailable(f"history ends {end:%Y-%m-%d}; too old for {ts.min():%Y-%m-%d}. Refresh the data")
        X = self._design(self._series, ts)
        if X[["lag_1d", "lag_7d", "level_1d"]].isna().any().any():
            raise PredictionUnavailable("history lacks the 1-day/7-day lags for some requested timestamps")
        raw = self._fitted.predict(X)
        prices = [round(float(p), 5) for p in raw]
        lower = [round(float(p) + self._q15_resid, 5) for p in raw]
        upper = [round(float(p) + self._q85_resid, 5) for p in raw]
        return PriceForecast(prices, self._best, self._mae, len(self._series), end, lower, upper)
