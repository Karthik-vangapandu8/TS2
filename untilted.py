import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
# Define the Streamlit app
def main():
    st.title("Service Consumption Prediction App")
    # User input for dataset
    st.subheader("Enter Dataset:")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write(df)

        # Check if required columns are present
        if {'billdservices', 'units', 'load', 'totservices'}.issubset(df.columns):
            # Prepare features (X) and target variable (y)
            X = df[['billdservices', 'units', 'load']]
            y = df['totservices']

            # Split the dataset into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train a Random Forest Regressor model
            model = RandomForestRegressor(random_state=42)
            model.fit(X_train, y_train)

            # User input for new data point
            st.subheader("Enter New Data Point:")
            billdservices = st.number_input("Enter billed services:")
            units = st.number_input("Enter units consumed:")
            load = st.number_input("Enter load:")

            # Predict total services for the new data point
            new_data_point = [[billdservices, units, load]]
            predicted_services = model.predict(new_data_point)

            # Display the predicted total services
            st.subheader("Predicted Total Services:")
            st.write(predicted_services[0])

            # Evaluate the model
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            st.subheader("Model Evaluation (Mean Absolute Error):")
            st.write(mae)
        else:
            st.error("Required columns (billdservices, units, load, totservices) are missing in the dataset.")

if __name__ == "__main__":
    main()
