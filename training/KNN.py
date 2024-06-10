from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np



def KNN (df, target_columns, feature_columns):
    X = df.loc[:, feature_columns]
    y = df.loc[:, target_columns].values

    X1 = df[feature_columns].columns[0]
    X2 = df[feature_columns].columns[1]
    # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)

    # Huấn luyện mô hình Linear Regression
    model = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)

    # Đưa ra dự đoán trên toàn bộ dữ liệu
    y_pred = model.predict(X_test)
    

    # Tính các chỉ số đánh giá (Mean Squared Error và R-squared)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return model, mse, r2, X_train, y_train, y_pred, X1, X2