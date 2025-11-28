from contextlib import asynccontextmanager
from fastapi import FastAPI, Form, Request, status
from fastapi.exceptions import HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd


BASE_PATH = Path(__file__).resolve().parent
YIELD_ASSETS_DIR = BASE_PATH / "yield_assets"
TEMPLATES_DIR = BASE_PATH / "templates"
logger = logging.getLogger(__name__)


def _load_artifacts(base_path: Path = BASE_PATH):
    """Load serialized artifacts and dataset required for inference."""
    required_files = [
        "best_model.pkl",
        "label_encoders.pkl",
        "median_values.pkl",
        "unique_values.pkl",
    ]
    missing = [file for file in required_files if not (base_path / file).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files: {', '.join(missing)}")

    model, feature_columns = joblib.load(base_path / "best_model.pkl")
    label_encoders = joblib.load(base_path / "label_encoders.pkl")
    unique_values = joblib.load(base_path / "unique_values.pkl")
    median_values = joblib.load(base_path / "median_values.pkl")
    df = pd.read_csv(base_path / "crop_yield.csv")

    return {
        "model": model,
        "feature_columns": feature_columns,
        "label_encoders": label_encoders,
        "unique_values": unique_values,
        "median_values": median_values,
        "df": df,
    }


def _load_crop_recommendation_assets(asset_dir: Path = YIELD_ASSETS_DIR):
    """Load classifiers and scalers for the crop recommendation form."""
    required_files = {
        "model": asset_dir / "model_first.pkl",
        "minmax": asset_dir / "minmaxscaler_first.pkl",
    }

    missing = [name for name, path in required_files.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing crop recommendation files: " + ", ".join(missing)
        )

    model = joblib.load(required_files["model"])
    minmax_scaler = joblib.load(required_files["minmax"])

    stand_scaler_path = asset_dir / "standscaler.pkl"
    standard_scaler = (
        joblib.load(stand_scaler_path) if stand_scaler_path.exists() else None
    )

    crop_lookup = {
        1: "rice",
        2: "maize",
        3: "jute",
        4: "cotton",
        5: "coconut",
        6: "papaya",
        7: "orange",
        8: "apple",
        9: "muskmelon",
        10: "watermelon",
        11: "grapes",
        12: "mango",
        13: "banana",
        14: "pomegranate",
        15: "lentil",
        16: "blackgram",
        17: "mungbean",
        18: "mothbeans",
        19: "pigeonpeas",
        20: "kidneybeans",
        21: "chickpea",
        22: "coffee",
        23: "wheat",
        24: "barley",
        25: "dates",
        26: "olive",
        27: "potato",
        28: "tomato",
    }

    return {
        "model": model,
        "minmax_scaler": minmax_scaler,
        "standard_scaler": standard_scaler,
        "crop_lookup": crop_lookup,
    }


try:
    _INITIAL_ARTIFACTS = _load_artifacts()
except Exception as exc:  # pragma: no cover
    logger.warning("Initial artifact load failed: %s", exc)
    _INITIAL_ARTIFACTS = None

try:
    _INITIAL_YIELD_ARTIFACTS = _load_crop_recommendation_assets()
except Exception as exc:  # pragma: no cover
    logger.warning("Initial crop recommendation load failed: %s", exc)
    _INITIAL_YIELD_ARTIFACTS = None


def _get_artifacts(app: FastAPI):
    artifacts = getattr(app.state, "artifacts", None)
    if artifacts is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifacts not loaded yet.",
        )
    return artifacts


def _get_crop_recommendation_assets(app: FastAPI):
    assets = getattr(app.state, "crop_recommendation", None)
    if assets is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Crop recommendation assets not available.",
        )
    return assets


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        if getattr(app.state, "artifacts", None) is None:
            app.state.artifacts = _load_artifacts()
            logger.info("Artifacts successfully loaded for prediction API.")
        if getattr(app.state, "crop_recommendation", None) is None:
            app.state.crop_recommendation = _load_crop_recommendation_assets()
            logger.info("Crop recommendation assets loaded.")
        yield
    except Exception as exc:  # pragma: no cover - startup failure logged
        logger.exception("Failed to load artifacts: %s", exc)
        app.state.artifacts = None
        app.state.crop_recommendation = None
        raise


app = FastAPI(title="Crop Yield Prediction API", lifespan=lifespan)
app.state.artifacts = _INITIAL_ARTIFACTS
app.state.crop_recommendation = _INITIAL_YIELD_ARTIFACTS
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    try:
        artifacts = _get_artifacts(app)
        unique_values = artifacts["unique_values"]
        context = {
            "request": request,
            "crops": unique_values["crops"],
            "states": unique_values["states"],
            "seasons": unique_values["seasons"],
        }
        return templates.TemplateResponse(request, "index.html", context)
    except HTTPException as exc:
        raise exc
    except Exception as exc:
        logger.exception("Error rendering home page: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to load form options.",
        )


@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    crop: str = Form(...),
    state: str = Form(...),
    season: str = Form(...),
    year: Optional[str] = Form(default=None),
    rainfall: Optional[str] = Form(default=None),
    fertilizer: Optional[str] = Form(default=None),
    pesticide: Optional[str] = Form(default=None),
):
    artifacts = _get_artifacts(app)
    unique_values = artifacts["unique_values"]

    def _error_response(message: str):
        context = {
            "request": request,
            "error": message,
            "crops": unique_values["crops"],
            "states": unique_values["states"],
            "seasons": unique_values["seasons"],
        }
        return templates.TemplateResponse(
            request, "index.html", context, status_code=status.HTTP_400_BAD_REQUEST
        )

    try:
        if not all([crop, state, season]):
            return _error_response("Please fill in all required fields.")

        crop = crop.strip().title()
        state = state.strip().title()
        season = season.strip()

        if crop not in unique_values["crops"]:
            return _error_response(f"Invalid crop selection: {crop}")
        if state not in unique_values["states"]:
            return _error_response(f"Invalid state selection: {state}")
        if season not in unique_values["seasons"]:
            return _error_response(f"Invalid season selection: {season}")

        median_values = artifacts["median_values"]
        input_data = pd.DataFrame(
            {
                "Crop": [crop],
                "State": [state],
                "Season": [season],
                "Crop_Year": [float(year) if year else median_values["Crop_Year"]],
                "Annual_Rainfall": [
                    float(rainfall) if rainfall else median_values["Annual_Rainfall"]
                ],
                "Fertilizer": [
                    float(fertilizer) if fertilizer else median_values["Fertilizer"]
                ],
                "Pesticide": [
                    float(pesticide) if pesticide else median_values["Pesticide"]
                ],
            }
        )

        label_encoders = artifacts["label_encoders"]
        for column, encoder in label_encoders.items():
            if column in input_data.columns:
                input_data[column] = encoder.transform(input_data[column])

        feature_columns = artifacts["feature_columns"]
        input_data = input_data[feature_columns]

        model = artifacts["model"]
        prediction = model.predict(input_data)
        pred_value = (
            float(prediction[0])
            if isinstance(prediction, (np.ndarray, list))
            else prediction
        )

        context = {
            "request": request,
            "crop": crop,
            "state": state,
            "season": season,
            "prediction": f"{pred_value:.2f}",
        }
        return templates.TemplateResponse(request, "result.html", context)

    except ValueError as exc:
        logger.warning("Validation error: %s", exc)
        return _error_response(str(exc))
    except Exception as exc:
        logger.exception("Unexpected prediction error: %s", exc)
        return _error_response("An unexpected error occurred. Please try again.")


@app.get("/crop-recommendation", response_class=HTMLResponse)
async def crop_recommendation_form(request: Request):
    _ = _get_crop_recommendation_assets(app)
    context = {"request": request}
    return templates.TemplateResponse(request, "crop_recommendation.html", context)


@app.post("/crop-recommendation/predict", response_class=HTMLResponse)
async def crop_recommendation_predict(
    request: Request,
    nitrogen: str = Form(...),
    phosphorus: str = Form(...),
    potassium: str = Form(...),
    temperature: str = Form(...),
    humidity: str = Form(...),
    ph: str = Form(...),
    rainfall: str = Form(...),
):
    assets = _get_crop_recommendation_assets(app)

    def _error(message: str):
        return templates.TemplateResponse(
            request,
            "crop_recommendation.html",
            {"request": request, "error": message},
            status_code=status.HTTP_400_BAD_REQUEST,
        )

    try:
        features = np.array(
            [
                float(nitrogen),
                float(phosphorus),
                float(potassium),
                float(temperature),
                float(humidity),
                float(ph),
                float(rainfall),
            ]
        ).reshape(1, -1)
    except ValueError:
        return _error("All values must be numeric.")

    minmax_scaler = assets["minmax_scaler"]
    standard_scaler = assets["standard_scaler"]
    model = assets["model"]
    crop_lookup = assets["crop_lookup"]

    transformed = minmax_scaler.transform(features)
    if standard_scaler is not None:
        transformed = standard_scaler.transform(transformed)

    prediction = model.predict(transformed)
    crop_code = int(prediction[0])
    crop_name = crop_lookup.get(crop_code, f"Crop #{crop_code}")

    context = {
        "request": request,
        "crop_name": crop_name.title(),
        "crop_code": crop_code,
        "inputs": {
            "Nitrogen": nitrogen,
            "Phosphorus": phosphorus,
            "Potassium": potassium,
            "Temperature": temperature,
            "Humidity": humidity,
            "pH": ph,
            "Rainfall": rainfall,
        },
    }
    return templates.TemplateResponse(
        request, "crop_recommendation_result.html", context
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
