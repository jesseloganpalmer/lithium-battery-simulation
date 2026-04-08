"""Defines the timeframe for the lithium battery simulation."""

from __future__ import annotations


def define_timeframe() -> list[float]:
    """Define the timeframe for the lithium battery simulation."""
    return [0, 3600]  # solve for 1 hour
