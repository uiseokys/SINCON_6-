"""Obsolete Public-derived calibration is blocked in this package."""
PUBLIC_STATION_WEIGHT = None
PUBLIC_LOG_LEVEL = None
def calibrate(*args, **kwargs):
    raise RuntimeError("Public-derived calibration is unavailable")
