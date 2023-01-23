from sklearn.model_selection import cross_val_score

N_SPLITS = 3

def get_score(estimator, X, y):
    full_scores = cross_val_score(
        estimator, X, y, scoring="neg_mean_squared_error", cv=N_SPLITS
    )
    return -full_scores.mean(), full_scores.std()