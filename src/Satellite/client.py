# src/Satellite/client.py
import ee
import os
import json
import pandas as pd
from datetime import datetime, timezone
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from src.utils.logger import logger

load_dotenv()

GEE_PROJECT = os.getenv("GEE_PROJECT_ID", "alternate-alpha-gee")
DATA_DIR = Path("data/satellite")
DATA_DIR.mkdir(parents=True, exist_ok=True)


def init_gee():
    try:
        ee.Initialize(project=GEE_PROJECT)
        logger.info(f"GEE initialized: {GEE_PROJECT}")
    except Exception as e:
        logger.error(f"GEE init failed: {e}")
        raise


def get_sentinel2_collection(bbox, start_date, end_date):
    region = ee.Geometry.Rectangle(bbox)
    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(region)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
        .select(["B4", "B8", "B11"])
    )
    return collection, region


def compute_occupancy(image, region, scale=10):
    nir = image.select("B8")
    car_threshold = 1500        # 0.15 reflectance * 10000
    car_mask = nir.lt(car_threshold)
    valid_mask = nir.gt(200).And(nir.lt(3000))  # exclude shadows + buildings

    car_pixels = car_mask.And(valid_mask).reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=region,
        scale=scale,
        maxPixels=1e6,
    )
    total_pixels = valid_mask.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=region,
        scale=scale,
        maxPixels=1e6,
    )
    occupancy = ee.Number(car_pixels.get("B8")).divide(
        ee.Number(total_pixels.get("B8")).max(1)
    )
    return occupancy.getInfo()


def fetch_occupancy_timeseries(location, start_date, end_date):
    init_gee()
    bbox = location["bbox"]
    name = location["name"]
    logger.info(f"Fetching: {name}")

    collection, region = get_sentinel2_collection(bbox, start_date, end_date)
    count = collection.size().getInfo()
    logger.info(f"  {count} valid images found")

    if count == 0:
        logger.warning(f"  No imagery for {name}")
        return []

    results = []
    image_list = collection.toList(collection.size())

    for i in range(count):
        try:
            image = ee.Image(image_list.get(i))
            date_ms = image.get("system:time_start").getInfo()
            date_str = datetime.fromtimestamp(date_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
            cloud_pct = image.get("CLOUDY_PIXEL_PERCENTAGE").getInfo()
            image_id = image.get("system:index").getInfo()
            occupancy = compute_occupancy(image, region)

            results.append({
                "date": date_str,
                "occupancy": float(occupancy) if occupancy is not None else None,
                "cloud_pct": float(cloud_pct),
                "image_id": image_id,
                "location": name,
            })
            logger.info(f"  {date_str}: occupancy={occupancy:.3f}, cloud={cloud_pct:.1f}%")

        except Exception as e:
            logger.warning(f"  Image {i} failed: {e}")
            continue

    return results


def fetch_all_locations(ticker, locations, start_date, end_date, save=True):
    all_results = []
    for loc in locations:
        results = fetch_occupancy_timeseries(loc, start_date, end_date)
        all_results.extend(results)

    if not all_results:
        logger.warning(f"No results for {ticker}")
        return pd.DataFrame()

    df = pd.DataFrame(all_results)
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["occupancy"])

    if save:
        raw_path = DATA_DIR / f"{ticker}_raw_{datetime.now().strftime('%Y%m%d')}.json"
        with open(raw_path, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"Raw saved: {raw_path}")

    daily = (
        df.groupby("date")["occupancy"]
        .agg(["mean", "std", "count"])
        .rename(columns={"mean": "occupancy_mean", "std": "occupancy_std", "count": "location_count"})
        .reset_index()
    )
    logger.info(f"{ticker}: {len(daily)} days of data")
    return daily


if __name__ == "__main__":
    ee.Initialize(project=GEE_PROJECT)
    print(ee.String("connected").getInfo())