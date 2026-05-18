import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import learning_curve

def generar_curvas_aprendizaje(X, y, estimator_type, cv):
    if estimator_type == 'logistic':
        est = LogisticRegression(max_iter=1000, random_state=42)
    else:
        est = DecisionTreeClassifier(max_depth=5, random_state=42)

    train_sizes_abs, train_scores, val_scores = learning_curve(
        est, X, y,
        train_sizes=np.linspace(0.2, 1.0, 5),
        cv=cv,
        scoring='accuracy',
    )

    return pd.DataFrame({
        'train_size':       train_sizes_abs,
        'train_score_mean': np.round(train_scores.mean(axis=1), 4),
        'train_score_std':  np.round(train_scores.std(axis=1), 4),
        'val_score_mean':   np.round(val_scores.mean(axis=1), 4),
        'val_score_std':    np.round(val_scores.std(axis=1), 4),
    })
#