from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def linear_regression(df, target_columns, feature_columns):
    X = df[feature_columns]
    y = df[target_columns]

    # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    # Huấn luyện mô hình Linear Regression
    model = LinearRegression().fit(X_train, y_train)

    # Đưa ra dự đoán trên toàn bộ dữ liệu
    y_pred = model.predict(X_test)

    # Tính các chỉ số đánh giá (Mean Squared Error và R-squared)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return model, mse, r2, X_test, y_test, y_pred
