# src/Satellite/locations.py
"""
Tickers: WMT (Walmart), TGT (Target), HD (Home Depot)
"""

RETAIL_LOCATIONS = {
    "WMT": [
        {
            "name": "Walmart Supercenter - Bentonville AR (HQ area)",
            "lat": 36.3729,
            "lon": -94.2088,
            "bbox": [-94.2105, 36.3715, -94.2070, 36.3743],
        },
        {
            "name": "Walmart Supercenter - Secaucus NJ",
            "lat": 40.7799,
            "lon": -74.0566,
            "bbox": [-74.0590, 40.7782, -74.0542, 40.7816],
        },
        {
            "name": "Walmart Supercenter - Lakewood CO",
            "lat": 39.7225,
            "lon": -105.0862,
            "bbox": [-105.0885, 39.7208, -105.0839, 39.7242],
        },
    ],
    "TGT": [
        {
            "name": "Target - Edina MN (near HQ)",
            "lat": 44.8893,
            "lon": -93.3499,
            "bbox": [-93.3520, 44.8876, -93.3478, 44.8910],
        },
        {
            "name": "Target - Newark CA",
            "lat": 37.5230,
            "lon": -122.0402,
            "bbox": [-122.0425, 37.5213, -122.0379, 37.5247],
        },
        {
            "name": "Target - Chicago IL (Addison)",
            "lat": 41.9319,
            "lon": -87.6601,
            "bbox": [-87.6624, 41.9302, -87.6578, 41.9336],
        },
    ],
    "HD": [
        {
            "name": "Home Depot - Atlanta GA (near HQ)",
            "lat": 33.7490,
            "lon": -84.3880,
            "bbox": [-84.3903, 33.7473, -84.3857, 33.7507],
        },
        {
            "name": "Home Depot - Burbank CA",
            "lat": 34.1808,
            "lon": -118.3089,
            "bbox": [-118.3112, 34.1791, -118.3066, 34.1825],
        },
        {
            "name": "Home Depot - Plano TX",
            "lat": 33.0137,
            "lon": -96.6989,
            "bbox": [-96.7012, 33.0120, -96.6966, 33.0154],
        },
    ],
}

EARNINGS_MONTHS = {
    "WMT": [2, 5, 8, 11],
    "TGT": [3, 6, 9, 12],
    "HD":  [2, 5, 8, 11],
}