from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import ee
import os
from dotenv import load_dotenv
import joblib
import numpy as np
import requests
from pathlib import Path 

env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
USE_BEARER_TOKEN = False
if GEMINI_API_KEY:
    if GEMINI_API_KEY.startswith("AQ."):
        USE_BEARER_TOKEN = True
        print("Gemini token loaded (appears to be an OAuth access token starting with 'AQ.').")
        print("Note: OAuth tokens are short-lived. For stable server usage create an API key in GCP Console.")
    else:
        print("Gemini API Key loaded successfully (will be sent as ?key=...).")
else:
    print("CRITICAL ERROR: Gemini API Key NOT FOUND! Check your .env file.")

app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)
try:
    ee.Initialize(project='isro-urban-heat-mitigation')
    print("Earth Engine Initialized Successfully!")
except Exception as e:
    print("Earth Engine initialization failed.")
    print("Enable the Earth Engine API for the Google Cloud project and wait a few minutes before retrying.")
    print("Open: https://console.developers.google.com/apis/api/earthengine.googleapis.com/overview?project=isro-urban-heat-mitigation")
    print(f"Details: {e}")

try:
    ml_model = joblib.load('heat_predictor.pkl')
except Exception as e:
    print(f"Model load error: {e}")

class SimulationRequest(BaseModel):
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float
    greenery_pct: float
    albedo_pct: float
    ventilation_pct: float
    water_pct: float
    smart_design_pct: float

@app.post("/api/heat-data-live")
def get_live_heat_data(req: SimulationRequest):
    """
    Dynamically fetches Sentinel-2 and Landsat 8 data for the drawn polygon,
    calculates real-world baselines, applies the ML model, and returns coordinates.
    """
    roi = ee.Geometry.BBox(req.min_lon, req.min_lat, req.max_lon, req.max_lat)
    
    sentinel = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(roi).filterDate('2025-01-01', '2025-12-31') \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10)).median()
        
    ndvi = sentinel.normalizedDifference(['B8', 'B4']).rename('NDVI') # Vegetation
    ndwi = sentinel.normalizedDifference(['B3', 'B8']).rename('NDWI') # Water
    ndbi = sentinel.normalizedDifference(['B11', 'B8']).rename('NDBI') # Built-up / Concrete
    
    landsat = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
        .filterBounds(roi).filterDate('2025-01-01', '2025-12-31') \
        .filter(ee.Filter.lt('CLOUD_COVER', 10)).median()
    
    lst_celsius = landsat.select('ST_B10').multiply(0.00341802).add(149.0).subtract(273.15).rename('LST')
    
    combined = ee.Image([ndvi, ndwi, ndbi, lst_celsius])
    sampled_points = combined.sample(region=roi, scale=30, numPixels=25, geometries=True).getInfo()
    
    heat_zones = []
    baseline_temps, ndvi_vals, ndwi_vals, ndbi_vals = [], [], [], []
    
    for feature in sampled_points['features']:
        coords = feature['geometry']['coordinates']
        props = feature['properties']
        
        if 'LST' not in props or 'NDVI' not in props:
            continue
            
        real_baseline_temp = props['LST']
        baseline_temps.append(real_baseline_temp)
        
        # Collect indices for baseline calculation
        if 'NDVI' in props: ndvi_vals.append(props['NDVI'])
        if 'NDWI' in props: ndwi_vals.append(props['NDWI'])
        if 'NDBI' in props: ndbi_vals.append(props['NDBI'])
        
        # Run ML Inference
        input_features = np.array([[req.greenery_pct, req.albedo_pct, req.ventilation_pct, req.water_pct, req.smart_design_pct]])
        predicted_temp = ml_model.predict(input_features)[0]
        cooling_delta = 42.0 - predicted_temp 
        
        heat_zones.append({
            "lon": coords[0],
            "lat": coords[1],
            "baseline_lst": round(real_baseline_temp, 2),
            "simulated_lst": round(real_baseline_temp - cooling_delta, 2)
        })
    
    # Calculate real-world averages
    avg_baseline = sum(baseline_temps) / len(baseline_temps) if baseline_temps else 42.0
    avg_simulated = sum([z['simulated_lst'] for z in heat_zones]) / len(heat_zones) if heat_zones else 42.0
    
    # Convert Indices to rough real-world percentages (0 to 100)
    avg_ndvi = sum(ndvi_vals) / len(ndvi_vals) if ndvi_vals else 0
    avg_ndwi = sum(ndwi_vals) / len(ndwi_vals) if ndwi_vals else 0
    avg_ndbi = sum(ndbi_vals) / len(ndbi_vals) if ndbi_vals else 0
    
    # Scaling math: NDVI of 0.5 is roughly 100% dense canopy. NDWI > 0 is water. NDBI > 0 is heavy concrete.
    exist_green = max(0, min(100, int((avg_ndvi / 0.5) * 100)))
    exist_blue = max(0, min(100, int((avg_ndwi / 0.3) * 100)))
    exist_built = max(0, min(100, int((avg_ndbi / 0.4) * 100)))

    return {
        "metadata": {
            "source": "LIVE: Landsat 8 / Sentinel-2",
            "baseline_lst": round(avg_baseline, 2),
            "predicted_avg_lst": round(avg_simulated, 2),
            "net_cooling_achieved": round(avg_baseline - avg_simulated, 2),
            "existing_conditions": {
                "green_pct": exist_green,
                "blue_pct": exist_blue,
                "built_pct": exist_built
            }
        },
        "spatial_data": heat_zones
    }

@app.get("/api/ai-synthesis")
def get_ai_synthesis(cooling: float, green: float, roof: float, vent: float, blue: float, smart: float):
    """
    Utilizes Gemini API to generate a professional urban planning brief 
    based on the physics-informed model outputs.
    """
    intervention_area_m2 = 50000 
    trees_needed = round((intervention_area_m2 * (green / 100)) / 30)
    roof_area = round(intervention_area_m2 * (roof / 100))
    pools_needed = round((intervention_area_m2 * (blue / 100)) / 1250, 1)

    prompt = f"""
    You are an expert ISRO geospatial analyst and urban planner. Review this localized simulation data for a 5-Hectare target area, derived from a physics-informed surface energy balance model:
    
    - Estimated Land Surface Temperature (LST) Reduction: -{cooling}°C
    - Intervention Weights Applied to Spatial Polygons: Urban Greening ({green}%), High-Albedo Roofs ({roof}%), Urban Ventilation ({vent}%), Blue Infrastructure ({blue}%), Smart Geometry ({smart}%).
    - Tangible Resource Requirements:
      * {trees_needed:,} mature canopy trees required (to alter Sentinel-2 NDVI)
      * {roof_area:,} sq. meters of high-reflectance surface coatings required
      * {pools_needed} standard municipal water body equivalents required (to alter NDWI)
    
    Provide a highly focused, 3-part strategic brief for city administrators. Format exactly like this, using plain text (no markdown, no asterisks, no bolding):
    
    HEAT STRESS IMPACT: (1 sentence on how this LST drop scientifically affects the local microclimate and energy grid)
    KEY DRIVERS: (1 sentence identifying the highest weighted interventions used and referencing their exact physical resource requirements)
    RECOMMENDATION: (1 short, punchy sentence on what the city council should authorize next based on this spatial data)
    """
    
    # Choose auth method based on the loaded key/token
    base_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent"
    headers = {'Content-Type': 'application/json'}
    if GEMINI_API_KEY and USE_BEARER_TOKEN:
        # If the env value is an OAuth access token (e.g. starts with 'AQ.'), use Bearer header
        url = base_url
        headers['Authorization'] = f"Bearer {GEMINI_API_KEY}"
    else:
        # Otherwise assume it's an API key usable via ?key=
        url = f"{base_url}?key={GEMINI_API_KEY}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}]
    }
    
    response = None
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=15)
        response.raise_for_status()
        response_data = response.json()

        analysis_text = None
        # Common GL API response shape
        if isinstance(response_data, dict):
            if 'candidates' in response_data and response_data['candidates']:
                try:
                    analysis_text = response_data['candidates'][0]['content']['parts'][0]['text']
                except Exception:
                    analysis_text = None
            # fallback for alternate response shapes
            if not analysis_text and 'output' in response_data:
                try:
                    analysis_text = response_data['output'][0]['content'][0]['text']
                except Exception:
                    analysis_text = None

        if analysis_text:
            return {"status": "success", "analysis": analysis_text}
        else:
            print("Unexpected API response structure:", response_data)
            return {"status": "error", "message": "AI Synthesis offline. Unexpected API response format (see server logs)."}

    except requests.exceptions.HTTPError as e:
        code = response.status_code if response is not None else 'N/A'
        print(f"HTTP Error {code}: {e}")
        if response is not None:
            print("Response body:", response.text)
        return {"status": "error", "message": "AI Synthesis offline. See server logs for API response."}
    except Exception as e:
        print(f"REST API Error: {e}")
        if response is not None:
            print("Response body:", response.text)
        return {"status": "error", "message": "AI Synthesis offline. Please check API key and network connection."}