# Heart Disease Risk Prediction System

This project provides a machine learning-based tool for predicting heart disease risk using CDC health metrics data.

## Project Structure

heart-disease-prediction/
├── data/ # Raw data directory
│ └── heart_2020_uncleaned.csv # Original dataset
├── model/ # Model training code
│ ├── train_model.py # Training script
│ ├── requirements.txt # Python dependencies
│ ├── heart_disease_model.pkl # Trained model (generated)
│ └── feature_columns.pkl # Feature names (generated)
├── app/ # Streamlit application
│ ├── app.py # Main application
│ ├── requirements.txt # App dependencies
│ └── assets/ # Static assets
└── README.md # Project documentation

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/heart-disease-prediction.git
cd heart-disease-prediction
```

2. Set up model training environment:
   cd model
   pip install -r requirements.txt

3. Set up Streamlit app environment:
   cd ../app
   pip install -r requirements.txt
