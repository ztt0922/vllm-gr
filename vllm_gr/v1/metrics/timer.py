# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Host interval timer; seconds travel unchanged to frontend histogram reporting."""

from __future__ import annotations

from time import perf_counter
from typing import Self


class WallTimer:
    """Start/stop or context timer, independent of any metrics SDK or registry."""

    def __init__(self) -> None:
        self._started: float | None = None
        self.seconds: float | None = None

    def start(self) -> Self:
        if self._started is not None:
            raise RuntimeError("Timer has already started")
        self._started = perf_counter()
        return self

    def stop(self) -> float:
        if self._started is None:
            raise RuntimeError("Timer has not started")
        if self.seconds is None:
            self.seconds = max(0.0, perf_counter() - self._started)
        return self.seconds

    def __enter__(self) -> Self:
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.stop()
