from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from mpl_toolkits.mplot3d import Axes3D



# def linear_regression(df, target_columns, feature_columns):
#     X = df[feature_columns]
#     y = df[target_columns]

#     # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

#     # Huấn luyện mô hình Linear Regression
#     model = LinearRegression().fit(X_train, y_train)

#     # Đưa ra dự đoán trên toàn bộ dữ liệu
#     y_pred = model.predict(X_test)

#     # Tính các chỉ số đánh giá (Mean Squared Error và R-squared)
#     mse = mean_squared_error(y_test, y_pred)
#     r2 = r2_score(y_test, y_pred)

#     return model, mse, r2, X_test, y_test, y_pred

def linear_regression(df, target_columns, feature_columns):
    # Extract independent and dependent variables
        X = df[feature_columns].values
        y = df[target_columns].values
        
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        
        # Fit the linear regression model
        reg = LinearRegression()
        reg.fit(X_train, y_train)
        
        # Create a variable for each dimension
        x = X[:, 0]
        y_feature = X[:, 1]
        z = df[target_columns].values
        
        # Define ranges for visualization
        x_range = np.linspace(min(x), max(x), 35)
        y_range = np.linspace(min(y_feature), max(y_feature), 35)
        x_range, y_range = np.meshgrid(x_range, y_range)
        
        # Predict price values using the linear regression model
        viz = np.array([x_range.flatten(), y_range.flatten()]).T
        predictions = reg.predict(viz)
        
        # Evaluate the model using the R^2 score
        r2 = reg.score(X_test, y_test)
        
        # Plotting the model for visualization
        plt.style.use('fivethirtyeight')
        
        # Initialize a matplotlib figure
        fig = plt.figure(figsize=(15, 6 ))
        
        axis1 = fig.add_subplot(131, projection='3d')
        axis2 = fig.add_subplot(132, projection='3d')
        axis3 = fig.add_subplot(133, projection='3d')
        
        axes = [axis1, axis2, axis3]
        
        for ax in axes:
            ax.plot(x, y_feature, z, color='k', zorder=10, linestyle='none', marker='o', alpha=0.1)
            ax.scatter(x_range.flatten(), y_range.flatten(), predictions, facecolor=(0, 0, 0, 0), s=20, edgecolor='#70b3f0')
            ax.set_xlabel('Area', fontsize=10, labelpad=10)
            ax.set_ylabel('Bedrooms', fontsize=10, labelpad=10)
            ax.set_zlabel('Prices', fontsize=10, labelpad=10)
            ax.locator_params(nbins=3, axis='x')
            ax.locator_params(nbins=3, axis='y')
        
        axis1.view_init(elev=50, azim=-60)
        axis2.view_init(elev=30, azim=15)
        axis3.view_init(elev=50, azim=60)
        
        fig.suptitle(f'Multi-Linear Regression Model Visualization (R2 = {r2:.2f})', fontsize=15, color='k')
        st.pyplot(plt.gcf())    
        return reg, predictions, r2
 
# Giả sử bạn đã có DataFrame 'housing', cột mục tiêu 'price', và các cột đặc trưng 'area' và 'bedrooms'
# df = pd.read_csv('housing.csv')
# target_column = 'price'
# feature_columns = ['area', 'bedrooms']

# Gọi hàm để trực quan hóa mô hình hồi quy tuyến tính
