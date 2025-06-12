# Laptop Price Predictor

A machine learning web application that predicts laptop prices based on their specifications. Built with Python, Streamlit, and scikit-learn.

## Features

- Predicts laptop prices based on various specifications
- Interactive web interface
- Supports multiple features:
  - Brand
  - Type
  - RAM
  - Weight
  - Touchscreen
  - IPS Display
  - Screen Size & Resolution
  - CPU Brand
  - Storage (HDD/SSD)
  - GPU Brand
  - Operating System

## Installation

1. Clone the repository
2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```
2. Open your browser and go to `http://localhost:8501`
3. Enter the laptop specifications
4. Click "Predict Price" to get the estimated price

## Project Structure

- `app.py`: Main Streamlit application
- `retrain_model.py`: Script to retrain the model
- `laptop_data.csv`: Dataset used for training
- `pipe.pkl`: Trained model pipeline
- `df.pkl`: Processed dataframe
- `templates/`: HTML templates
- `static/`: CSS and other static files

## Model Details

- Uses Random Forest Regressor
- Features engineered from raw specifications
- Preprocessed with OneHotEncoder for categorical variables
- Logarithmic transformation applied to price values

## Technologies Used

- Python
- Streamlit
- scikit-learn
- pandas
- numpy
- Flask
