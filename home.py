import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn import tree
from matplotlib.colors import ListedColormap
import numpy as np
from streamlit_option_menu import option_menu
# Function to load CSV file with specified encoding
@st.cache_data
def load_csv(file, encoding='utf-8'):
    df = pd.read_csv(file, encoding=encoding)
    return df

# Function to select target and features
def select_columns(df):
    st.subheader("Select Target Column:")
    target_column = st.selectbox("Select the target column:", df.columns)

    st.subheader("Select Feature Columns:")
    feature_columns = st.multiselect("Select the feature columns:", df.columns)

    return target_column, feature_columns

# Function to process data
def process_data(df, target_column, feature_columns):
    # Process your data here
    target_data = df[target_column]  # Select the target column
    feature_data = df[feature_columns]  # Select the feature columns
    
    return target_data, feature_data
    
def app():
    st.title("CSV Uploader and Column Selector")
    st.write("Upload your CSV file:")
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])

    if uploaded_file is not None:
        df = load_csv(uploaded_file, encoding='latin1')

        with st.expander("View Full Data", expanded=True):
            st.write(df.head(5))

        target_column, feature_columns = select_columns(df)

        st.write("You selected the following target column:")
        st.write(target_column)

        st.write("You selected the following feature columns:")
        st.write(feature_columns)

        # Process data
        target_data, feature_data = process_data(df, target_column, feature_columns)

        # Optionally, you can display processed data
        st.write("Processed target data:")
        st.write(target_data)

        st.write("Processed feature data:")
        st.write(feature_data)
        model_type = st.selectbox("Select Model Type:", ["Linear Regression", "Decision Tree", "KNN", "Logistic Regression"])

        # Train model
        if st.button("Train Model"):
            st.write("Training model...")
                # if model_type == "Linear Regression":
                #     model, mse, r2, X_test, y_test, y_pred   = linear_tr.linear_regression(df, target_column, feature_columns)
                #     st.write("Model trained successfully!")
                #     st.write("Mean Squared Error:", mse)
                #     st.write("R-squared:", r2)  
                #     linear_tr.plot_2d_model(model, X_test, y_test, y_pred)

                #     # # Vẽ biểu đồ tương ứng với số chiều của dữ liệu
                #     # if feature_columns == 1:
                #     #     linear_tr.plot_2d_scatter(X_test, y_test, y_pred)
                #     # elif feature_columns >= 2:
                #     #     linear_tr.plot_3d_scatter(X_test, y_test, y_pred)

                # elif model_type == "Decision Tree":
                #     model, mse, r2, X_test, y_test, y_pred = knn_train.train_model(df, target_column, feature_columns, model_type)

                #     # Display metrics
                #     st.write("Mean Squared Error:", mse)
                #     st.write("R-squared:", r2)

                #     # Plot Decision Tree
                #     st.write("Decision Tree Visualization:")
                #     plt.figure(figsize=(20,10))
                #     tree.plot_tree(model, feature_names=feature_columns, filled=True)
                #     st.pyplot(plt)
                # elif model_type == "KNN":
                #     model, mse, r2, X_test, y_test, y_pred = knn_train.train_model_knn(df, target_column, feature_columns)
                # elif model_type == "Logistic Regression":
                #     model, mse, r2, X_test, y_test, y_pred = trlog.logistic_regression(df, target_column, feature_columns)
                #     plot_predictions(y_test, y_pred)
