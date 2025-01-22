import streamlit as st
import pandas as pd
from src.data.loader import load_data, preprocess
from src.models.trainer import train_models, evaluate_models

# Page config
st.set_page_config(page_title="Iris Classifier", layout="wide")

def main():
    st.title("Iris Species Classification")
    
    # Load and show data
    df = load_data()
    if st.checkbox("Show raw data"):
        st.dataframe(df)
    
    # Preprocess data
    X_train, X_test, y_train, y_test = preprocess(df)
    
    # Train models
    models = train_models(X_train, y_train)
    
    # Model evaluation
    st.subheader("Model Evaluation")
    eval_results = evaluate_models(models, X_test, y_test)
    st.dataframe(pd.DataFrame(eval_results))
    
    # Prediction interface
    st.subheader("Make Prediction")
    with st.form("prediction_form"):
        sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0)
        sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0)
        petal_length = st.number_input("Petal Length (cm)", min_value=0.0)
        petal_width = st.number_input("Petal Width (cm)", min_value=0.0)
        
        model_choice = st.selectbox("Select Model", list(models.keys()))
        
        if st.form_submit_button("Predict"):
            # Create DataFrame with proper feature names
            input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                                    columns=X_train.columns)
            model = models[model_choice]
            prediction = model.predict(input_data)[0]
            st.success(f"Predicted Species: {prediction}")

if __name__ == "__main__":
    main()