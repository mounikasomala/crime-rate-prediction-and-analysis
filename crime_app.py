import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def load_data(file_path="C:/Users/mouni/OneDrive/Desktop/Cleaned_Crime_data.csv"):
    """Loads crime data from a CSV file."""
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"Error: Could not find the file at '{file_path}'. Please check the file path.")
        return pd.DataFrame()

def train_model(crime_data):
    """Trains a linear regression model for each state and crime type."""
    models = {}
    for state in crime_data['Area_Name'].unique():
        state_data = crime_data[crime_data['Area_Name'] == state]
        for crime_type in state_data['Crime_Type'].unique():
            filtered_data = state_data[state_data['Crime_Type'] == crime_type]
            # Drop rows with NaN values
            filtered_data = filtered_data.dropna(subset=['Crime_count'])
            if len(filtered_data) > 1:  # At least 2 data points are required
                X = filtered_data[['Year']].values
                y = filtered_data['Crime_count'].values
                model = LinearRegression()
                model.fit(X, y)
                models[(state, crime_type)] = model
    return models

def predict_crime(model, year):
    """Predicts the crime count for a given year using the trained model."""
    return round(model.predict([[year]])[0])  # Round the predicted value to nearest integer

def calculate_r2_score(model, X, y):
    """Calculates the R-squared score of the model."""
    y_pred = model.predict(X)
    return r2_score(y, y_pred)

def main():
    st.title("Crime Data Analysis and Prediction")

    # Load crime data from CSV
    crime_data = load_data()

    if crime_data.empty:
        return

    # Check for necessary columns
    required_columns = ['Area_Name', 'Year', 'Crime_Type', 'Crime_count']
    if not all(col in crime_data.columns for col in required_columns):
        st.error("Error: Missing required columns in the data. Please check your CSV file.")
        return

    # Train models
    models = train_model(crime_data)

    # Remove 'Andaman & Nicobar Islands' from the list of states
    states = sorted([state for state in crime_data['Area_Name'].unique() if state != 'Andaman & Nicobar Islands'])

    # Get years available in the dataset
    years = sorted(crime_data['Year'].unique())
    valid_years = [year for year in range(min(years), 2016)]  # Restrict years to 2001–2015

    # User input
    selected_state = st.selectbox("Select State", states)
    selected_year = st.selectbox("Select Year", valid_years)

    if selected_state and selected_year:
        submit_button = st.button("Submit")

        if submit_button:
            st.write(f"**Crime Data Analysis for {selected_state}:**")

            # Filter data for the selected state
            state_data = crime_data[crime_data['Area_Name'] == selected_state]

            if selected_year <= 2010:
                # Analyze data for the selected year
                filtered_data = state_data[state_data['Year'] == selected_year]
                if not filtered_data.empty:
                    # Create a bar chart for the selected year
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.barplot(x="Crime_Type", y="Crime_count", data=filtered_data, color='seagreen')
                    plt.xlabel('Crime Type')
                    plt.ylabel('Number of Crimes')
                    plt.title(f"Crime Data for {selected_state} in {selected_year}")
                    st.pyplot(fig)
                else:
                    st.warning(f"No data available for {selected_state} in {selected_year}.")
            else:
                # Predict crime for the selected year
                st.write(f"**Predicted Crime Rate for {selected_state} in {selected_year}:**")
                predicted_data = []
                for crime_type in state_data['Crime_Type'].unique():
                    model = models.get((selected_state, crime_type))
                    if model:
                        predicted_count = predict_crime(model, selected_year)
                        predicted_data.append({"Crime_Type": crime_type, "Crime_count": predicted_count})
                
                if predicted_data:
                    predicted_df = pd.DataFrame(predicted_data)
                    
                    # Display predicted values and accuracy above the graph
                    st.write("### Predicted Values with Accuracy")
                    for _, row in predicted_df.iterrows():
                        # Get the model for this crime type
                        model = models.get((selected_state, row['Crime_Type']))
                        if model:
                            # Get the actual data for R2 calculation
                            state_data = crime_data[crime_data['Area_Name'] == selected_state]
                            filtered_data = state_data[state_data['Crime_Type'] == row['Crime_Type']]
                            filtered_data = filtered_data.dropna(subset=['Crime_count'])
                            if len(filtered_data) > 1:
                                X = filtered_data[['Year']].values
                                y = filtered_data['Crime_count'].values
                                r2 = calculate_r2_score(model, X, y)
                                accuracy = max(0, min(100, round(r2 * 100, 1)))  # Convert to percentage and clamp between 0-100
                                st.write(f"- **{row['Crime_Type']}**: {row['Crime_count']} crimes (Accuracy: {accuracy}%)")

                    # Create a bar chart for the predicted values
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.barplot(x="Crime_Type", y="Crime_count", data=predicted_df, color='brown')
                    plt.xlabel('Crime Type')
                    plt.ylabel('Predicted Number of Crimes')
                    plt.title(f"Predicted Crime Data for {selected_state} in {selected_year}")

                    # Annotate values above the bars
                    for i, row in predicted_df.iterrows():
                        ax.text(i, row['Crime_count'], f"{row['Crime_count']}", ha='center', va='bottom')
                    
                    st.pyplot(fig)

if __name__ == "__main__":
    main()