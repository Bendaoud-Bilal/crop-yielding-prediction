# Crop Yield Prediction Web App

This repository contains a FastAPI-based web application for predicting crop yields based on various factors such as crop type, state, season, and other agricultural data. The model used for prediction is a machine learning model that was trained using historical crop yield data.

The application provides users with an interactive form where they can enter relevant information, and the model will predict the yield in metric tons per hectare.

## Features

- Predict crop yield based on various inputs such as crop name, state, season, year, rainfall, fertilizer usage, and pesticide usage.
- Optional inputs are supported (crop year, rainfall, fertilizer, and pesticide usage). If not provided, default values based on median values from the dataset are used.
- User-friendly form with error handling to ensure proper inputs are provided.
- Displays predicted crop yield in metric tons per hectare.

## Prerequisites

To run this project locally, you'll need to have Python 3.x installed. Additionally, you will need the following Python libraries:

- FastAPI
- Uvicorn
- NumPy
- Pandas
- scikit-learn
- joblib
- Jinja2 / Bootstrap (for frontend templating & styling)

## Training Progress

![training_progress](training_progress.png)

## Installation & Local Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/Bendaoud-Bilal/crop-yielding-prediction.git
   cd crop-yielding-prediction
   ```

2. **Create a dedicated virtual environment** (recommended)

   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\Activate.ps1
   # macOS / Linux
   source .venv/bin/activate
   ```

3. **Install dependencies and (optionally) retrain the model artifacts**

   ```bash
   pip install -r requirements.txt
   python main.py  # run only if you need to regenerate the pickled artifacts
   ```

4. **Verify artifact files** — the API requires the following files in the repo root:

   - `best_model.pkl`
   - `label_encoders.pkl`
   - `unique_values.pkl`
   - `median_values.pkl`
   - `crop_yield.csv`

5. **Run the FastAPI app via Uvicorn**

   ```bash
   uvicorn app:app --reload
   ```

6. **Open the UI** at `http://127.0.0.1:8000/` and submit predictions through the form.

### Running the automated tests

```bash
pytest tests/test_app.py
```

The tests perform lightweight smoke checks against the FastAPI routes using the serialized artifacts.

## File Structure

```
crop-yielding-prediction/
├── app.py                     # FastAPI application entrypoint
├── main.py                    # Training script that regenerates model artifacts
├── crop_yield.csv             # Dataset used for training the model
├── best_model.pkl             # Pretrained model used for predictions
├── label_encoders.pkl         # Label encoders used to encode categorical features
├── unique_values.pkl          # Dropdown values for crops, states, and seasons
├── median_values.pkl          # Median fallback values for optional inputs
├── requirements.txt           # Python dependencies
├── templates/                 # HTML templates for the UI
│   ├── index.html             # Prediction form
│   └── result.html            # Prediction output page
├── tests/                     # Lightweight FastAPI smoke tests
│   └── test_app.py
└── .venv/ (optional)          # Local virtual environment
```

## Usage

1. Visit the main page and fill in the form with the following details:

   - **Crop Name** (Required)
   - **State Name** (Required)
   - **Season** (Required)
   - **Crop Year** (Optional)
   - **Annual Rainfall** (Optional)
   - **Fertilizer Use** (Optional)
   - **Pesticide Use** (Optional)

2. Click on **Predict Yield** to get the predicted crop yield.

3. The prediction will be displayed in metric tons per hectare.

## Model Details

The machine learning model used in this application is a **Gradient Boosting Regressor**, which was trained using historical crop yield data. The features used in the model include:

- Crop
- State
- Season
- Crop Year
- Annual Rainfall
- Fertilizer Usage
- Pesticide Usage

The model predicts the crop yield based on these features.

## Troubleshooting

If you encounter issues such as missing files or errors in prediction:

- Ensure that all required model files (`best_model.pkl`, `label_encoders.pkl`, `unique_values.pkl`, and `median_values.pkl`) are in place.
- Ensure that you are using the correct versions of the required libraries.

## Contributing

If you'd like to contribute to this project, feel free to fork the repository, create a branch, and submit a pull request with your changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
