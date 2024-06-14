from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from mpl_toolkits.mplot3d import Axes3D

def linear_regression(df, target_columns, feature_columns):
    # Extract independent and dependent variables
        X = df[feature_columns].values
        y = df[target_columns].values
        
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        
        # Fit the linear regression model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Create a variable for each dimension
        x = X[:, 0]
        y_feature = X[:, 1]
        z = df[target_columns].values
        
        # Define ranges for visualization
        x_range = np.linspace(min(x), max(x), 35)
        y_range = np.linspace(min(y_feature), max(y_feature), 35)
        x_range, y_range = np.meshgrid(x_range, y_range)
        
        # Predict price values using the linear modelression model
        viz = np.array([x_range.flatten(), y_range.flatten()]).T
        y_pred = model.predict(viz)
        
        # Evaluate the model using the R^2 score
        r2 = model.score(X_test, y_test)
        mse = mean_squared_error(y_test, model.predict(X_test) )
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
            ax.scatter(x_range.flatten(), y_range.flatten(), y_pred, facecolor=(0, 0, 0, 0), s=20, edgecolor='#70b3f0')
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
        return model, y_pred, r2, y_test, mse
 


#fig, ax = plt.subplots(1, 2, figsize=(18,4))

# amount_val = df['Amount'].values
# time_val = df['Time'].values

# sns.distplot(amount_val, ax=ax[0], color='r')
# ax[0].set_title('Distribution of Transaction Amount', fontsize=14)
# ax[0].set_xlim([min(amount_val), max(amount_val)])

# sns.distplot(time_val, ax=ax[1], color='b')
# ax[1].set_title('Distribution of Transaction Time', fontsize=14)
# ax[1].set_xlim([min(time_val), max(time_val)])