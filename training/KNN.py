from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
# import category_encoders as ce

def KNN(df, target_column, feature_columns, k_values):
    X = df[feature_columns]
    y = df[target_column]
    
    X_train, X_text, y_train, y_text = train_test_split(X, y, test_size=0.3, random_state=17)

    cv_scores, holdout_scores = [], []

    for k in k_values:
        knn_pipe = Pipeline([
            ("scaler", StandardScaler()), 
            ("knn", KNeighborsClassifier(n_neighbors=k))
        ])
        # danh sách các điểm số chính xác được tính toán thông qua kỹ thuật cross-validation
        cv_score = np.mean(cross_val_score(knn_pipe, X_train, y_train, cv=5))
        knn_pipe.fit(X_train, y_train)
        holdout_score = accuracy_score(y_text, knn_pipe.predict(X_text))
        
        cv_scores.append(cv_score)
        holdout_scores.append(holdout_score)
        
    print(holdout_score)
    return k_values, cv_scores, holdout_scores


    # X = df[feature_columns]
    # y = df[target_columns]
    
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=17)
    # cols = X_train.columns
    # # Feature Scaling
    # scaler = StandardScaler()

    # X_train = scaler.fit_transform(X_train)

    # X_test = scaler.transform(X_test)
    # X_train = pd.DataFrame(X_train, columns=[cols])
    # X_test = pd.DataFrame(X_test, columns=[cols])
    # # instantiate the model
    # model = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
    #                  metric_params=None, n_jobs=None, n_neighbors=5, p=2,
    #                  weights='uniform')
    # model.fit(X_train, y_train)
    
    # y_pred = model.predict(X_test)
    # as_test = accuracy_score(y_test, y_pred)
    # y_pred_train = model.predict(X_train)
    # as_train = accuracy_score(y_train, y_pred_train)
    # cm = confusion_matrix(y_test, y_pred)

    # print('Confusion matrix\n\n', cm)

    # print('\nTrue Positives(TP) = ', cm[0,0])

    # print('\nTrue Negatives(TN) = ', cm[1,1])

    # print('\nFalse Positives(FP) = ', cm[0,1])

    # print('\nFalse Negatives(FN) = ', cm[1,0])
    
    # #Classification metrices 
    # print(classification_report(y_test, y_pred))
    
    # TP = cm[0,0]
    # TN = cm[1,1]
    # FP = cm[0,1]
    # FN = cm[1,0]
    
    # classification_accuracy = (TP + TN) / float(TP + TN + FP + FN)

    # classification_error = (FP + FN) / float(TP + TN + FP + FN)
    
    # precision = TP / float(TP + FP)

    # print('Precision : {0:0.4f}'.format(precision))
    
    # recall = TP / float(TP + FN)

    # print('Recall or Sensitivity : {0:0.4f}'.format(recall))
    
    # # y_pred_prob = model.predict_proba(X_test)[0:10, 1]
    # # print(y_pred_prob)
    # y_pred_1 = model.predict_proba(X_test)[:, 1]
    # # print("retterhgt",y_pred_1)
    # # print('Classification accuracy : {0:0.4f}'.format(classification_accuracy))
    # #plot
    
    # return X_test, y_test, model, as_test, as_train, y_pred, cm, classification_accuracy, classification_error, y_pred_1
      
    
    
    
    
    
    
    
    
    # X = df.loc[:, feature_columns]
    # y = df.loc[:, target_columns].values

    # X1 = df[feature_columns].columns[0]
    # X2 = df[feature_columns].columns[1]
    # # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)

    # # Huấn luyện mô hình Linear Regression
    # model = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)

    # # Đưa ra dự đoán trên toàn bộ dữ liệu
    # y_pred = model.predict(X_test)
    

    # # Tính các chỉ số đánh giá (Mean Squared Error và R-squared)
    # mse = mean_squared_error(y_test, y_pred)
    # r2 = r2_score(y_test, y_pred)

    # return model, mse, r2, X_train, y_train, y_pred, X1, X2