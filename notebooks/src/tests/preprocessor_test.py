
import numpy as np
import scipy as sp
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.svm import SVR
import xgboost as xgb
from sklearn.pipeline import make_pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression, LassoCV

def preprocessor_test(preprocessor):


    l1_regulatization = [0.1, 0.4, 0.5, 0.55, 0.6, 0.7, 0.9, 0.95, 0.99, 1]

    inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)

    lr_pipeline = make_pipeline(
        preprocessor,
        #SelectFromModel(LassoCV()),
        TransformedTargetRegressor(
            regressor=LinearRegression(), 
            func=np.log10, 
            inverse_func=sp.special.exp10
        )
    )

    ridge_pipeline = make_pipeline(
        preprocessor,
        #SelectFromModel(LassoCV()),
        TransformedTargetRegressor(
            regressor=RidgeCV(
                #alphas =l2_regulatization, 
                cv=inner_cv, 
                #store_cv_values=True,
                scoring = "neg_mean_squared_error", 
                ), 
            func=np.log1p, 
            inverse_func=np.expm1
        ) 
    )

    lasso_pipeline = make_pipeline(
        preprocessor,
        TransformedTargetRegressor(
            regressor=LassoCV(
                    alphas=l1_regulatization,
                    cv=inner_cv
                    ), 
                func=np.log1p, 
                inverse_func=np.expm1
        )
    )
    elasticnet_pipeline = make_pipeline(
        preprocessor,
        #SelectFromModel(LassoCV()),
        TransformedTargetRegressor(
            regressor=ElasticNetCV(
                    cv=inner_cv,
                    l1_ratio=l1_regulatization
                ), 
                func=np.log1p, 
                inverse_func=np.expm1
        )
    )
    svr_pipeline = make_pipeline(
        preprocessor,
        #SelectFromModel(LassoCV()),
        TransformedTargetRegressor(
            regressor= SVR(
                kernel='poly',
                degree=2
            ),
            func=np.log1p, 
            inverse_func=np.expm1
        )
    )

    rf_pipeline = make_pipeline(
        preprocessor,
        TransformedTargetRegressor(
            regressor= RandomForestRegressor(),
            func=np.log1p, 
            inverse_func=np.expm1
        )
    ) 

    #xgb_study = joblib.load("hyperparameter_tuning\study_xgb.pkl")
    xgb_pipeline = make_pipeline(
        preprocessor,
        TransformedTargetRegressor(
            regressor=xgb.XGBRegressor(), #(**xgb_study.best_params),
            func=np.log1p, 
            inverse_func=np.expm1
        )
    )

    estimators = [
            #('lr', lr_pipeline), 
            ('ridge', ridge_pipeline),
            ('lasso', lasso_pipeline),
            ('enet', elasticnet_pipeline), 
            ('svr', svr_pipeline),
            ('xgb',xgb_pipeline),
            ('rf', rf_pipeline)
    ]

    stacking_regressor = StackingRegressor(
        estimators=estimators,
        final_estimator= ElasticNetCV(
                    l1_ratio=l1_regulatization,
                    cv=inner_cv
        )
    )
    return stacking_regressor