from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np # linear algebra
import seaborn as sns # statistical data visualization
import category_encoders as ce

def RandomForestCF (df, target_columns, feature_columns):
    X = df.drop(feature_columns, axis=1)
    y = df[target_columns]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)
    
    encoder = ce.OrdinalEncoder(cols = x_train.columns )
    model = RandomForestClassifier(n_estimators=200, random_state=0)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    ass = accuracy_score(y_test, y_pred)
    feature_scores = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    return  model, X_test, y_test, y_pred, ass, feature_scores