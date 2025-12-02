from __future__ import annotations

import logging
from typing import Any
from typing import Dict
from typing import List
from typing import Type

from .base import BaseRunner

log = logging.getLogger("app.runners.timeseries")


TIMESERIES_TASKS = {"time-series-forecasting"}


class TimeSeriesForecastingRunner(BaseRunner):
    def load(self) -> int:
        # No heavy model; mark as loaded
        self._loaded = True
        return 0

    def predict(
        self, inputs: Dict[str, Any], options: Dict[str, Any]
    ) -> Dict[str, Any]:
        series = inputs.get("series")
        if not isinstance(series, list) or not series:
            return {"forecast": [], "quantiles": {}}
        import numpy as np

        arr = np.array(series, dtype=float)
        horizon = int(options.get("horizon", 5))
        method = str(options.get("method", "persistence"))
        forecast: List[float] = []
        if method == "mean":
            mu = float(np.mean(arr))
            forecast = [mu for _ in range(horizon)]
        else:
            # persistence: repeat last value
            last = float(arr[-1])
            forecast = [last for _ in range(horizon)]

        # Simple empirical std for quantiles assuming normality
        std = float(np.std(arr))
        q = {
            "p10": [float(v - 1.2816 * std) for v in forecast],
            "p50": forecast,
            "p90": [float(v + 1.2816 * std) for v in forecast],
        }
        return {"forecast": forecast, "quantiles": q}


_TASK_TO_RUNNER: Dict[str, Type[BaseRunner]] = {
    "time-series-forecasting": TimeSeriesForecastingRunner,
}


def runner_for_task(task: str) -> Type[BaseRunner]:
    return _TASK_TO_RUNNER[task]


__all__ = [
    "TIMESERIES_TASKS",
    "TimeSeriesForecastingRunner",
    "runner_for_task",
]
