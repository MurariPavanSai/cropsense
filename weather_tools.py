# File: weather_tools.py
 
import requests
import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry
from strands import tool
import os
import re
import json
import numpy as np
from httpx import Client
from datetime import datetime, timedelta
 
# Tool 1: Get latitude and longitude from ZIP code using OpenWeatherMap API
@tool
def get_location_by_zip(zip_code: str, country_code: str = "IN") -> dict:
    """
    Fetch the latitude, longitude, and city name for a given ZIP code using the OpenWeatherMap Geocoding API.
 
    Use this tool when you need geographic coordinates for a location based on a ZIP/PIN code, typically for weather or location-based queries.
 
    Args:
        zip_code: The ZIP or PIN code (e.g., "507115" for Indian locations). Must be a string.
        country_code: The ISO country code (default: "IN" for India).
 
    Returns:
        A dictionary with keys: "latitude" (float), "longitude" (float), "city" (str). If failed, includes "error" (str).
    """
    api_key = "76a0c6e97a0199f64856f78941ccc701"
    if not api_key:
        return {"error": "OpenWeatherMap API key not set in environment variable OPENWEATHERMAP_API_KEY"}
    if not re.match(r"^\d{6}$", zip_code):  # Validate Indian PIN code (6 digits)
        return {"error": "Invalid PIN code. Must be a 6-digit number."}
   
    url = "http://api.openweathermap.org/geo/1.0/zip"
    params = {
        "zip": f"{zip_code},{country_code}",
        "appid": api_key
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        print(f"DEBUG: Fetched coordinates for {zip_code}: {data}")
        return {
            "latitude": data.get("lat"),
            "longitude": data.get("lon"),
            "city": data.get("name")
        }
    except requests.RequestException as e:
        return {"error": f"Failed to fetch coordinates: {str(e)}"}
 
# Tool 2: Fetch and analyze weather data from Open-Meteo API
@tool
def get_weather_analysis(latitude: float, longitude: float) -> dict:
    """
    Fetch and summarize historical weather data from Open-Meteo API for the past 30 days, including temperature, precipitation, relative humidity, and soil moisture.
 
    Use this tool after obtaining coordinates to get a climate summary with min/max values and averages.
 
    Args:
        latitude: The latitude of the location (float).
        longitude: The longitude of the location (float).
 
    Returns:
        A dictionary with weather summary: "Temp_min" (float), "Temp_max" (float), "Rain" (float, total cm), "RH_avg" (float, %), "Soil Moisture" (str, categorized as Low, Medium, or High). If failed, includes "error" (str).
    """
    # Setup Open-Meteo API client with cache and retry
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)  # Cache for 1 hour
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)
 
    # Define API parameters for past 30 days
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": [
            "temperature_2m",
            "soil_moisture_0_to_1cm",
            "soil_moisture_3_to_9cm",
            "soil_moisture_9_to_27cm",
            "soil_moisture_27_to_81cm",
            "precipitation",
            "relative_humidity_2m"
        ],
        "past_days": 30
    }
 
    try:
        # Make API call
        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]
        print(f"DEBUG: Fetched weather data for lat={latitude}, lon={longitude}")
 
        # Process hourly data
        hourly = response.Hourly()
        hourly_data = {
            "date": pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left"
            ),
            "temperature_2m": hourly.Variables(0).ValuesAsNumpy(),
            "soil_moisture_0_to_1cm": hourly.Variables(1).ValuesAsNumpy(),
            "soil_moisture_3_to_9cm": hourly.Variables(2).ValuesAsNumpy(),
            "soil_moisture_9_to_27cm": hourly.Variables(3).ValuesAsNumpy(),
            "soil_moisture_27_to_81cm": hourly.Variables(4).ValuesAsNumpy(),
            "precipitation": hourly.Variables(5).ValuesAsNumpy(),
            "relative_humidity_2m": hourly.Variables(6).ValuesAsNumpy()
        }
        df = pd.DataFrame(data=hourly_data)
        print(f"DEBUG: Hourly data shape: {df.shape}")
        print(f"DEBUG: Sample data:\n{df.head()}")
 
        # Compute average soil moisture across depths and time
        soil_columns = ["soil_moisture_0_to_1cm", "soil_moisture_3_to_9cm", "soil_moisture_9_to_27cm", "soil_moisture_27_to_81cm"]
        avg_soil_moisture = float(df[soil_columns].mean().mean())
        if avg_soil_moisture < 0.2:
            soil_category = "Low"
        elif avg_soil_moisture < 0.4:
            soil_category = "Medium"
        else:
            soil_category = "High"
        print(f"DEBUG: Average soil moisture: {avg_soil_moisture}, Categorized: {soil_category}")
 
        # Compute summary metrics
        total_precip_mm = float(df["precipitation"].sum())
        summary = {
            "Temp_min": round(float(df["temperature_2m"].min()), 2),
            "Temp_max": round(float(df["temperature_2m"].max()), 2),
            "Rain": round(total_precip_mm / 10, 2),  # Total precipitation in cm
            "RH_avg": round(float(df["relative_humidity_2m"].mean()), 2),
            "Soil Moisture": soil_category  # Categorized as Low, Medium, or High
        }
        return summary
    except Exception as e:
        return {"error": f"Failed to fetch weather data: {str(e)}"}
 
# Tool 3: Main weather analysis function to combine ZIP lookup and weather data
@tool
def analyze_weather(zip_code: str, country_code: str = "IN") -> dict:
    """
    Analyze weather for a ZIP/PIN code by first fetching coordinates and then getting the weather summary.
 
    This is the primary tool for weather queries based on PIN code. It returns a comprehensive summary.
 
    Args:
        zip_code: The ZIP or PIN code (e.g., "507115").
        country_code: The ISO country code (default: "IN").
 
    Returns:
        A dictionary with "city" (str) and weather metrics: "Temp_min", "Temp_max", "Rain", "RH_avg", "Soil Moisture". If failed, includes "error" (str).
    """
    # Step 1: Get coordinates
    location = get_location_by_zip(zip_code, country_code)
    if "error" in location:
        return {"error": location["error"]}
   
    # Step 2: Get weather analysis
    summary = get_weather_analysis(location["latitude"], location["longitude"])
    if "error" in summary:
        return {"error": summary["error"]}
   
    summary["city"] = location["city"]
    print(f"DEBUG: Final summary for {zip_code}: {summary}")
    return summary
 
@tool
def get_soil_type(latitude: float, longitude: float) -> dict:
    """
    Fetch the most probable soil type at the queried location using OpenEPI Soil API.
 
    Args:
        latitude: The latitude of the location (float).
        longitude: The longitude of the location (float).
 
    Returns:
        A dictionary containing soil type information.
    """
    with Client() as client:
        # Get the most probable soil type at the queried location
        response = client.get(
            url="https://api.openepi.io/soil/type/summary",
            params={"min_lon": longitude-0.01, "max_lon": longitude+0.01, "min_lat": latitude-0.01, "max_lat": latitude+0.01},
        )
 
        json_data = response.json()
        print(json_data)
        return json_data
 
