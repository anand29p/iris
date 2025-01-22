# Iris Classification App


A Streamlit application for classifying Iris flower species using machine learning.

## Project Structure

```
.
├── src/
│   ├── data/
│   │   └── loader.py        # Data loading and preprocessing
│   ├── models/
│   │   └── trainer.py       # Model training and evaluation
│   └── app.py               # Streamlit UI
├── main.py                  # Application entry point
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

```bash
streamlit run main.py
```

## Modules

- **Data Module**: Handles data loading and preprocessing
- **Models Module**: Manages model training and evaluation
- **App Module**: Contains the Streamlit UI and main application logic