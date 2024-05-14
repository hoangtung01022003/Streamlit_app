import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn import tree
from matplotlib.colors import ListedColormap
import numpy as np
from streamlit_option_menu import option_menu
import tempfile
import base64
# Function to load CSV file with specified encoding
@st.cache_data
def load_csv(file, encoding='utf-8'):
    df = pd.read_csv(file, encoding=encoding)
    return df

# def select_columns(df):
#     if 'df' not in st.session_state:
#         st.warning("Hãy tải dữ liệu trước khi thực hiện thao tác này.")
#         return

#     df = st.session_state.df
#     # Select column
#     selected_column = st.selectbox("Select Column:", df.columns)
    
#     # Confirm action
#     if st.button("Xác nhận"):
#         action = st.radio("Select Action:", ["Xoá", "Lấy giá trị trung bình"])
        
#         if action == "Xoá":
#             df.drop(columns=[selected_column], inplace=True)
#             st.write("Đã xoá cột", selected_column)
#         elif action == "Lấy giá trị trung bình":
#             mean_value = df[selected_column].mean()
#             st.write(f"Giá trị trung bình của cột {selected_column}: {mean_value}")

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

def app():
    st.title("CSV Uploader and Column Selector")
    st.write("Upload your CSV file:")
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])

    if uploaded_file is not None:
        df = load_csv(uploaded_file, encoding='latin1')
        if 'processed_df' in st.session_state:
            with st.expander("View Full Data (Processed)", expanded=True):
                st.write(st.session_state.processed_df.head(5))
        # Save processed data to a temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            df.to_csv(temp_file.name, index=False)

            # Create a link to download the CSV file
            st.markdown(save_data_clean(temp_file.name, 'Download processed data'), unsafe_allow_html=True)

            # Delete the temporary file after download link is created
            temp_file.close()
            total_rows = st.session_state.processed_df.shape[0]
            st.write(f"Tổng số lượng hàng: {total_rows}")
            null_counts = st.session_state.processed_df.isnull().sum()
            st.write("Số lượng giá trị null trong mỗi cột:")
            st.write(null_counts)
        else:
            with st.expander("View Full Data", expanded=True):
                st.write(df.head(5))
            total_rows = df.shape[0]
            st.write(f"Tổng số lượng hàn: {total_rows}")
            null_counts = df.isnull().sum()
            st.write("Số lượng giá trị null trong mỗi cột:")
            st.write(null_counts)
        cleaning_type = st.selectbox("Select Cleaning Type:", ["Xử lý giá trị thiếu", "Loại bỏ dữ liệu nhiễu"])
            
        # Train modelg

        if cleaning_type == "Xử lý giá trị thiếu":           
            selected_columns = st.multiselect("Select Column:", df.columns)
            action = st.radio("Select Action:", ["Xoá dữ liệu null", "Lấy giá trị trung bình"])

            # Confirm action
            if st.button("Xác nhận"):
                if action == "Xoá dữ liệu null":
                    df.dropna(subset=selected_columns, inplace=True)
                    st.write("Đã xoá dữ liệu null trong các cột", selected_columns)
                elif action == "Lấy giá trị trung bình":
                    for column in selected_columns:
                        mean_value = df[column].mean()
                        st.write(f"Giá trị trung bình của cột {column}: {mean_value}")
                st.session_state.processed_df = df
                # Select column
        
                    

