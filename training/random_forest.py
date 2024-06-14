from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import seaborn as sns 
import category_encoders as ce
from sklearn.metrics import confusion_matrix
import streamlit as st


def RandomForestCF (df, target_columns, feature_columns):
    X = df.drop(feature_columns, axis=1)
    y = df[target_columns]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)
    #chuyển đổi biến phân loại string thành int
    encoder = ce.OrdinalEncoder(cols = X_train.columns )
    X_train = encoder.fit_transform(X_train)
    X_test = encoder.transform(X_test)
    model = RandomForestClassifier(n_estimators=1000, random_state=0)
    model.fit(X_train, y_train)
    ass = accuracy_score(y_test, y_pred)
    y_pred = model.predict(X_test)
    
    feature_scores = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    return  model, X_test, y_test, y_pred, ass, feature_scores

def plot_confusion_matrix(y_test, y_pred, classes):
    conf_mat = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    st.pyplot(plt.gcf())