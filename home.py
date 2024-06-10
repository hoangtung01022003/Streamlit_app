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
import training.logistic_regression as train_logistic
import training.KNN as train_KNN
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import seaborn as sns # statistical data visualization
from scipy.special import expit
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier



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
        model_type = st.selectbox("Select Model Type:", ["Linear Regression", "Logistic Regression", "Decision Tree", "Random Forest", "KNN"])

        # Train model
        if st.button("Train Model"):
            st.write("Training model...")
            if model_type == "Linear Regression":
                model, mse, r2, X_test, y_test, y_pred = train_linear.linear_regression(df, target_columns, feature_columns)
                st.write("Model trained successfully!")
                st.write("Mean Squared Error:", mse)
                st.write("R-squared:", r2)  
                plt.figure(figsize=(10, 6))
                plt.scatter(y_test, y_pred, color='blue', label='Thực tế vs. Dự đoán')
                plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red', label='Đường chéo')
                plt.xlabel('Thực tế')
                plt.ylabel('Dự đoán')
                plt.title('Biểu đồ Scatter giữa Thực tế và Dự đoán')
                plt.legend()
                st.pyplot(plt.gcf())

                # Vẽ biểu đồ histogram của sai số dự đoán
                plt.figure(figsize=(10, 6))
                errors = y_test - y_pred
                plt.hist(errors, bins=20, color='green', alpha=0.7)
                plt.xlabel('Sai số dự đoán')
                plt.ylabel('Số lượng')
                plt.title('Biểu đồ Histogram của Sai số dự đoán')

                # Hiển thị biểu đồ trong ứng dụng Streamlit
                st.pyplot(plt.gcf())

            if model_type == "Logistic Regression":
                model,mse, r2, X_test, y_test, y_pred= train_logistic.logistic_regression(df, target_columns, feature_columns)
                st.write("Model trained successfully!")
                st.write("Mean Squared Error:", mse)
                st.write("R-squared:", r2)  
                plt.figure(figsize=(10, 6))
                plt.scatter(y_test, y_pred, color='blue', label='Thực tế vs. Dự đoán')
                plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red', label='Đường chéo')
                plt.xlabel('Thực tế')
                plt.ylabel('Dự đoán')
                plt.title('Biểu đồ Scatter giữa Thực tế và Dự đoán')
                plt.legend()
                st.pyplot(plt.gcf())
                
                # Vẽ biểu đồ histogram của sai số dự đoán
                plt.figure(figsize=(10, 6))
                errors = y_test - y_pred
                plt.hist(errors, bins=20, color='green', alpha=0.7)
                plt.xlabel('Sai số dự đoán')
                plt.ylabel('Số lượng')
                plt.title('Biểu đồ Histogram của Sai số dự đoán')

                # Hiển thị biểu đồ trong ứng dụng Streamlit
                st.pyplot(plt.gcf())

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
                # st.write(feature_scores)
                # sns.barplot(x=feature_scores, y=feature_scores.index)
                # plt.xlabel('Feature Importance Score')
                # plt.ylabel('Features')
                # plt.title("Visualizing Important Features")
                # st.pyplot(plt.gcf())

            if model_type == "KNN":
                model, mse, r2, X_train, y_train, y_pred, X0, X1 = train_KNN.KNN(df, target_columns, feature_columns)
                st.write("Model trained successfully!")
                
                cmap_light = ListedColormap(['orange', 'cyan', 'cornflowerblue'])
                cmap_bold = ListedColormap(['darkorange', 'c', 'darkblue'])
                h = .02  # step size in the mesh

                # Plot the decision boundary. For that, we will assign a color to each
                # point in the mesh [x_min, x_max]x[y_min, y_max].
                x_min, x_max = X_train.loc[:, X0].values.min() - 1, X_train.loc[:, X0].values.max() + 1
                y_min, y_max = X_train.loc[:, X1].values.min() - 1, X_train.loc[:, X1].values.max() + 1

                xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                    np.arange(y_min, y_max, h))

                Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

                # Put the result into a color plot
                Z = Z.reshape(xx.shape)
                plt.figure()
                plt.pcolormesh(xx, yy, Z, cmap=cmap_light, shading='nearest')

                # Plot also the training points
                plt.scatter(X_train.loc[:, X0].values,
                            X_train.loc[:, X1].values,
                            c=y_train,
                            cmap=cmap_bold,
                            edgecolor='k',
                            s=20)
                plt.xlim(xx.min(), xx.max())
                plt.ylim(yy.min(), yy.max())
                plt.title("3-Class classification (k = 5)")
               
                st.pyplot(plt.gcf())