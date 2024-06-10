import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn import tree
from matplotlib.colors import ListedColormap
import numpy as np
from streamlit_option_menu import option_menu
import training.linear_regression as train_linear
import training.decision_tree as train_tree

import training.random_forest as train_random
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import seaborn as sns # statistical data visualization



# Function to load CSV file with specified encoding
@st.cache_data
def load_csv(file, encoding='utf-8'):
    df = pd.read_csv(file, encoding=encoding)
    return df

# Function to select target and features
def select_columns(df):
    st.subheader("Select Target Column:")
    target_columns = st.selectbox("Select the target column:", df.columns)

    st.subheader("Select Feature Columns:")
    feature_columns = st.multiselect("Select the feature columns:", df.columns)

    return target_columns, feature_columns

# Function to process data
def process_data(df, target_columns, feature_columns):
    # Process your data here
    target_data = df[target_columns]  # Select the target column
    feature_data = df[feature_columns]  # Select the feature columns
    
    return target_data, feature_data
    
def plot_predictions(y_actual, y_pred):
    plt.scatter(y_actual, y_pred, color='blue')
    plt.plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], 'k--', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title("Thuật toán ")
    plt.show()

def home():
    st.title("CSV Uploader and Column Selector")
    st.write("Upload your CSV file:")
    uploaded_file1 = st.file_uploader("Choose a CSV file", type=['csv'])

    if uploaded_file1 is not None:
        df = load_csv(uploaded_file1, encoding='latin1')

        with st.expander("View Full Data", expanded=True):
            st.write(df.head(5))

        target_columns, feature_columns = select_columns(df)

        st.write("You selected the following target column:")
        st.write(target_columns)

        st.write("You selected the following feature columns:")
        st.write(feature_columns)

        # Process data
        target_data, feature_data = process_data(df, target_columns, feature_columns)

        # Optionally, you can display processed data
        st.write("Processed target data:")
        st.write(target_data)

        st.write("Processed feature data:")
        st.write(feature_data)
        model_type = st.selectbox("Select Model Type:", ["Linear Regression", "Decision Tree", "Random Forest", "Logistic regression", "KNN"])

        # Train model
        if st.button("Train Model"):
            st.write("Training model...")
            if model_type == "Linear Regression":
                reg, predictions, r2 = train_linear.linear_regression(df, target_columns, feature_columns)
                st.write("Model trained successfully!")
                # st.write("Mean Squared Error:", mse)
                st.write("R-squared:", r2)  
                # plt.figure(figsize=(10, 6))
                # plt.scatter(y_test, y_pred, color='blue', label='Thực tế vs. Dự đoán')
                # plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red', label='Đường đường chéo')
                # plt.xlabel('Thực tế')
                # plt.ylabel('Dự đoán')
                # plt.title('Biểu đồ Scatter giữa Thực tế và Dự đoán')
                # plt.legend()
                # st.pyplot(plt.gcf())

                # Vẽ biểu đồ histogram của sai số dự đoán
                # plt.figure(figsize=(10, 6))
                # errors = y_test - y_pred
                # plt.hist(errors, bins=20, color='green', alpha=0.7)
                # plt.xlabel('Sai số dự đoán')
                # plt.ylabel('Số lượng')
                # plt.title('Biểu đồ Histogram của Sai số dự đoán')

                # # Hiển thị biểu đồ trong ứng dụng Streamlit
                # st.pyplot(plt.gcf())
            if model_type == "Decision Tree":
                model, cm,  mse, r2, X_test, y_test, y_pred = train_tree.decision_tree(df, target_columns, feature_columns)
                st.write("Model trained successfully!")
                st.write("Mean Squared Error:", mse)
                st.write("R-squared:", r2)  
                st.write("Dự đoán:", cm)
                disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap=plt.cm.Blues, normalize='true')
                plt.title('Confusion Matrix')
                st.pyplot(plt.gcf())
            if model_type == "Random Forest":
                model, X_test, y_test, y_pred, ass, feature_scores = train_random.RandomForestCF(df, target_columns, feature_columns)
                st.write("Model trained successfully!")
                st.write('Model accuracy score with doors variable removed : {0:0.4f}'. format(ass))
                st.write(feature_scores)
                sns.barplot(x=feature_scores, y=feature_scores.index)
                plt.xlabel('Feature Importance Score')
                plt.ylabel('Features')
                plt.title("Visualizing Important Features")
                st.pyplot(plt.gcf())
                train_random.plot_confusion_matrix(y_test, y_pred, classes=model.classes_)