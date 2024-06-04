import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
def main():
    st.title("Service Consumption Prediction App")
    st.subheader("Enter Dataset:")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write(df)
        if {'billdservices', 'units', 'load', 'totservices'}.issubset(df.columns):
            X = df[['billdservices', 'units', 'load']]
            y = df['totservices']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = RandomForestRegressor(random_state=42)
            model.fit(X_train, y_train)
            st.subheader("Enter New Data Point:")
            billdservices = st.number_input("Enter billed services:")
            units = st.number_input("Enter units consumed:")
            load = st.number_input("Enter load:")
            new_data_point = [[billdservices, units, load]]
            predicted_services = model.predict(new_data_point)
            st.subheader("Predicted Total Services:")
            st.write(predicted_services[0])
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            st.subheader("Model Evaluation (Mean Absolute Error):")
            st.write(mae)
        else:
            st.error("Required columns (billdservices, units, load, totservices) are missing in the dataset.")
if __name__ == "__main__":
    main()
