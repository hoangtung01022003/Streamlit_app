import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn import tree
from matplotlib.colors import ListedColormap
import numpy as np
from streamlit_option_menu import option_menu
import tempfile
import base64
from sklearn.preprocessing import LabelEncoder

# Function to load CSV file with specified encoding
@st.cache_data
def load_csv(file, encoding='utf-8'):
    df = pd.read_csv(file, encoding=encoding)
    return df


def process_data(df, target_column):
    # Process your data here
    target_data = df[target_column]  # Select the target column
    
    return target_data
# Function save data
def save_data_clean(bin_file, file_label='File', button_label='Download'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{file_label}.csv">{button_label}</a>'
    return href

def select_columns(df):
    st.subheader("Select Feature Columns:")
    selected_columns = st.multiselect("Select columns:", df.columns)
    return selected_columns


def app():
    st.title("CSV Uploader and Column Selector")
    st.write("Upload your CSV file:")
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    if uploaded_file is not None:
        df = load_csv(uploaded_file, encoding='latin1')
        if 'processed_df' in st.session_state:
            with st.expander("View Full Data (Processed)", expanded=True):
                st.write(st.session_state.processed_df)
        # Save processed data to a temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            df.to_csv(temp_file.name, index=False)

            # Create a link to download the CSV file
            st.markdown(save_data_clean(temp_file.name, 'Download processed data'), unsafe_allow_html=True)

            # Delete the temporary file after download link is created
            temp_file.close()
            total_rows = st.session_state.processed_df.shape[0]
            st.write(f"Total quantity of rows: {total_rows}")
            null_counts = st.session_state.processed_df.isnull().sum()
            st.write("Number of null values ​​in each column:")
            st.write(null_counts)
        else:
            with st.expander("View Full Data", expanded=True):
                st.write(df)
            total_rows = df.shape[0]
            st.write(f"Total quantity of rows: {total_rows}")
            null_counts = df.isnull().sum()
            st.write("Number of null values ​​in each column:")
            st.write(null_counts)
        cleaning_type = st.selectbox("Select Cleaning Type:", ["Handling missing values", "Remove columns", "Change data"])
        selected_columns = select_columns(df)
        # Train modelg
        if 'processed_df' not in st.session_state:
            st.session_state.processed_df = df.copy()
        if cleaning_type is not None:
            df_original = st.session_state.processed_df.copy()
            
            if cleaning_type == "Handling missing values":           
                action = st.radio("Select Action:", ["Delete null data", "Take the average values"])

                # Confirm action
                if st.button("Confirm"):
                    if action == "Delete null data":
                        df_original.dropna(subset=selected_columns, inplace=True)
                        st.write("Delete null data success", selected_columns)
                    elif action == "Take the average values":
                        for column in selected_columns:
                            mean_value = df_original[column].mean()
                            st.write(f"Giá trị trung bình của cột {column}: {mean_value}")
            
                    # Select column
            elif cleaning_type == "Remove columns":
                if st.button("Confirm"):
                    df_original.drop(columns=selected_columns, inplace=True)
                    st.write("Column deleted successfully", selected_columns)
                
            elif cleaning_type == "Change data":
                if st.button("Convert"):
                    label_encoders = {}
                    for column in selected_columns:
                        le = LabelEncoder()
                        df_original[column] = le.fit_transform(df_original[column])
                        label_encoders[column] = le
                        st.write(f"Converted columns {column}")
                    
            st.session_state.processed_df = df_original