 
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from strands import tool
import json
import re
 
CROP_DATA_PATH = "crop_data.csv"
 
# Soil type mapping for consistency
SOIL_MAPPING = {
    "regur": "black/regur",
    "black": "black/regur",
    "clay": "clay",
    "alluvial": "alluvial",
    # Add more mappings as needed
}
 
def parse_range(value):
    """Parses a 'min–max' range string into average or single value."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        value = value.replace("–", "-")  # Normalize en dash to hyphen
        nums = re.findall(r"-?\d+\.?\d*", value)
        if len(nums) == 2:
            return (float(nums[0]) + float(nums[1])) / 2
        elif len(nums) == 1:
            return float(nums[0])
    return np.nan
 
def load_crop_data():
    """Load and preprocess crop data from CSV."""
    try:
        df = pd.read_csv(CROP_DATA_PATH)
        # Parse ranges from crop_data.csv
        df["Temperature_min"] = df["Temp (°C)"].apply(lambda x: float(re.findall(r"-?\d+\.?\d*", x.replace("–", "-"))[0]))
        df["Temperature_max"] = df["Temp (°C)"].apply(lambda x: float(re.findall(r"-?\d+\.?\d*", x.replace("–", "-"))[1]))
        df["Precipitation"] = df["Rain (cm)"].apply(parse_range)
        df["Humidity"] = df["RH (%)"].apply(parse_range)
        numeric_cols = ["Temperature_min", "Temperature_max", "Precipitation", "Humidity"]
        scaler = MinMaxScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        return df, scaler
    except Exception as e:
        print(f"Error loading crop data: {str(e)}")
        return pd.DataFrame(), None
 
crop_df, scaler = load_crop_data()
 
def climate_to_vector(climate_json, scaler):
    """Convert climate JSON to a scaled numeric vector."""
    if not isinstance(climate_json, dict):
        print("Error: climate_json is not a dictionary")
        return None, None
    temp_range = climate_json.get("Temperature (°C)", "0-0")
    temp_vals = re.findall(r"-?\d+\.?\d*", temp_range.replace("–", "-"))
    temp_min = float(temp_vals[0]) if temp_vals else 0.0
    temp_max = float(temp_vals[1]) if len(temp_vals) > 1 else temp_min
    prec = parse_range(climate_json.get("Precipitation (cm)", "0"))
    rh = parse_range(climate_json.get("Relative Humidity (%)", "0"))
    soil_moist = climate_json.get("Soil Moisture", "Medium")
    moisture_map = {"Low": 0.0, "Medium": 0.5, "High": 1.0, "Low–Medium": 0.25, "Mod–High": 0.75}
    soil_moist_num = moisture_map.get(soil_moist, 0.5)
    # Use DataFrame to preserve feature names
    arr = pd.DataFrame(
        [[temp_min, temp_max, prec, rh]],
        columns=["Temperature_min", "Temperature_max", "Precipitation", "Humidity"]
    )
    arr_scaled = scaler.transform(arr) if scaler is not None else arr
    return arr_scaled[0], soil_moist_num
 
def soil_type_match(user_soils, crop_soils):
    """Compute soil type match score."""
    if not isinstance(crop_soils, str):
        return 0.0
    user_soils = [SOIL_MAPPING.get(s.lower(), s.lower()) for s in user_soils]
    crop_soils = [s.lower().strip() for s in crop_soils.split(",")]
    return sum(1 for s in user_soils if s in crop_soils) / max(len(crop_soils), 1)
 
@tool
def recommend_crops(climate_json: dict, top_k: int = 5) -> dict:
    """Recommend best crops based on vector similarity."""
    print(f"Received climate_json: {climate_json}")  # Debug input
    if not isinstance(climate_json, dict):
        return {"error": "Invalid input: climate_json must be a dictionary"}
    if crop_df.empty or scaler is None:
        return {"error": "Crop data or scaler not initialized"}
    try:
        user_vec, soil_moist_num = climate_to_vector(climate_json, scaler)
        if user_vec is None:
            return {"error": "Failed to parse climate data"}
       
        crop_numeric = crop_df[["Temperature_min", "Temperature_max", "Precipitation", "Humidity"]].values
        sim = cosine_similarity([user_vec], crop_numeric)[0]
       
        indian_soil = climate_json.get("Indian Soil Type", [])
        fao_soil = climate_json.get("FAO/WRB Soil Type", [])
       
        soil_match = []
        for _, row in crop_df.iterrows():
            score = 0.0
            if isinstance(row["Indian Soil Type"], str):
                score += 0.25 * soil_type_match(indian_soil, row["Indian Soil Type"])
            if isinstance(row["FAO/WRB Soil Type"], str):
                score += 0.25 * soil_type_match(fao_soil, row["FAO/WRB Soil Type"])
            soil_match.append(score)
        soil_match = np.array(soil_match)
 
        final_score = (0.7 * sim) + (0.3 * soil_match)
        crop_df["similarity"] = final_score
        top_crops = crop_df.sort_values("similarity", ascending=False).head(top_k)
        return {
            "recommended_crops": top_crops[["Crop", "similarity"]].to_dict(orient="records")
        }
    except Exception as e:
        return {"error": str(e)}
 
 
