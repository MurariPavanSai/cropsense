# app.py
 
import re
import json
import logging
from flask import Flask, render_template, request, jsonify
from strands import Agent
from weather_tools import get_location_by_zip, get_weather_analysis, analyze_weather, get_soil_type
from crop_tools import recommend_crops
 
# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
 
# Combine all tools into one agent for effective chaining and multi-tool usage
tools = [
    get_location_by_zip,
    get_weather_analysis,
    analyze_weather,
    get_soil_type,
    recommend_crops
]
 
agent = Agent(tools=tools)
 
app = Flask(__name__)
 
@app.route('/')
def index():
    return render_template('index.html')
 
@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    pin_code = data.get('pinCode')
    crop = data.get('crop')
 
    if not re.match(r"^\d{6}$", pin_code):
        return jsonify({"error": "Please enter a valid 6-digit PIN code."}), 400
    elif not crop.strip():
        return jsonify({"error": "Please enter a crop name."}), 400
 
    # Construct the query to output JSON
    query = f"""
Analyze the weather for PIN code {pin_code} in India over the past 30 days.
Fetch coordinates using get_location_by_zip, then weather data using get_weather_analysis or analyze_weather,
and soil type using get_soil_type.
 
Summarize in an intermediate JSON format, including:
- "temperature": "min–max °C" (e.g., "20–30 °C")
- "precipitation": total as string (e.g., "150 cm" or "150–300 cm")
- "humidity": average as string (e.g., "60–80 %")
- "indian_soil_types": array of predominant Indian soil types from the soil API (e.g., ["Clay", "Alluvial"])
- "soil_moisture": "Low", "Medium", or "High"
 
Then, use this intermediate JSON with the recommend_crops tool to get top 5 crop recommendations with their scores.
 
Finally, assess if '{crop}' is a good choice: find its rank and score in the top 5 (if not in top 5, rank="N/A", score=0).
If score <= 0.5, include "alternatives" as array of top 3 recommended crop names; else omit "alternatives".
 
Output ONLY the final JSON in this exact format, no other text:
{{
  "weather": {{
    "temperature": "from intermediate",
    "precipitation": "from intermediate",
    "humidity": "from intermediate"
  }},
  "soil": {{
    "indianTypes": [array from intermediate indian_soil_types],
    "moisture": "from intermediate soil_moisture"
  }},
  "recommendedCrops": [
    {{"name": "crop1", "score": score1}},
    ...
  ],
  "cropSuitability": {{
    "cropName": "{crop}",
    "score": score,
    "rank": rank,
    "alternatives": ["top3"] // only if score <=0.5
   
  }},
  "insights": "Insights from the analysis that why the crop is suitable or not and also what can be better for the crop"
}}
"""
 
    try:
        # Call the agent with the query
        response = agent(query)
        logger.debug(f"Agent response type: {type(response)}")
        logger.debug(f"Agent response attributes: {dir(response)}")
 
        # Handle AgentResult object
        json_data = None
        if hasattr(response, 'content'):
            json_data = response.content
        elif hasattr(response, 'data'):
            json_data = response.data
        elif hasattr(response, 'result'):
            json_data = response.result
        else:
            # Try converting to string as a fallback
            try:
                json_data = str(response)
                logger.debug(f"Converted AgentResult to string: {json_data}")
            except Exception as e:
                logger.error(f"Failed to convert AgentResult to string: {str(e)}")
                raise ValueError("Cannot extract JSON from AgentResult: no known attributes or string conversion failed")
 
        # Process the response
        if isinstance(json_data, str):
            logger.debug(f"Raw JSON string: {json_data}")
            try:
                json_data = json.loads(json_data)  # Parse string to dict
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {str(e)}")
                return jsonify({"error": f"Invalid JSON format from agent: {str(e)}"}), 500
        elif isinstance(json_data, dict):
            logger.debug("Response is already a dict")
        else:
            logger.error(f"Unexpected response type: {type(json_data)}")
            return jsonify({"error": f"Unexpected response type from agent: {type(json_data)}"}), 500
 
        # Validate the JSON structure
        required_keys = ["weather", "soil", "recommendedCrops", "cropSuitability"]
        if not all(key in json_data for key in required_keys):
            logger.error(f"Invalid JSON structure: missing keys. Got: {json_data.keys()}")
            return jsonify({"error": "Invalid JSON structure from agent: missing required keys"}), 500
 
        logger.debug(f"Final JSON data: {json_data}")
        return jsonify(json_data)
 
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        logger.debug(f"Agent response (in error): {response}")
        return jsonify({"error": f"Error during analysis: {str(e)}"}), 500
 
if __name__ == '__main__':
    app.run(debug=True)
 
