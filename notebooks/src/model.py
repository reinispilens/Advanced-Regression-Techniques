
# General purpose
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

import math
from itertools import zip_longest


from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.compose import TransformedTargetRegressor

# Modeling libraries
from sklearn.model_selection import RandomizedSearchCV, KFold, RepeatedKFold, ShuffleSplit, cross_validate, cross_val_predict, cross_val_score, train_test_split
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
import xgboost as xgb

# Preprocessing libraries
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, RobustScaler, StandardScaler, MinMaxScaler, QuantileTransformer, PowerTransformer, PolynomialFeatures, StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.compose import TransformedTargetRegressor

# Modeling libraries
from sklearn.linear_model import LinearRegression, LassoCV


#redefine subsets according to preprocessing needs
discrete = ['YearBuilt', 'YearRemodAdd','BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
            'BedroomAbvGr', 'KitchenAbvGr','TotRmsAbvGrd','Fireplaces', 'GarageYrBlt','GarageCars', 
            'MoSold', 'YrSold', 'OverallQual', 'OverallCond']

continuous = ['LotFrontage', 'LotArea','MasVnrArea','BsmtFinSF1',  'BsmtFinSF2', 'BsmtUnfSF', 
              'TotalBsmtSF','1stFlrSF', '2ndFlrSF', 'LowQualFinSF','GrLivArea', 'GarageArea',  
              'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch','ScreenPorch', 'PoolArea', 
              'MiscVal']

nominal = ['MSSubClass','MSZoning', 'Street', 'Alley',  'LandContour','LotConfig',   'Neighborhood', 
           'Condition1', 'Condition2', 'BldgType', 'HouseStyle','RoofStyle','RoofMatl', 'Exterior1st',
           'Exterior2nd', 'MasVnrType',  'Foundation','Heating',  'CentralAir',  'GarageType','MiscFeature',
           'SaleType', 'SaleCondition']

ordinal = ['LotShape', 'LandSlope', 'Utilities',   'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond',
           'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2','HeatingQC', 'Electrical','KitchenQual', 
           'Functional','FireplaceQu', 'GarageFinish', 'GarageQual','GarageCond','PavedDrive', 
           'PoolQC', 'Fence']

# arrange categorical variable in categorical order to improve performance of linear models
LotShape = ['Reg', 'IR1', 'IR2', 'IR3']
LandSlope = ['Gtl', 'Mod', 'Sev']
Utilities = ['AllPub', 'NoSewr', 'NoSeWa', 'ELO']
ExterQual = ['Ex', 'Gd', 'TA', 'Fa', 'Po']
ExterCond = ['Ex', 'Gd', 'TA', 'Fa', 'Po']
BsmtQual = ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA']
BsmtCond = ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA']
BsmtExposure = ['Gd', 'Av', 'Mn', 'No', 'NA']
BsmtFinType1 = ['GLQ', 'ALQ', 'BLQ', 'Rec', 'LwQ', 'Unf', 'NA']
BsmtFinType2 = ['GLQ', 'ALQ', 'BLQ', 'Rec', 'LwQ', 'Unf', 'NA']
HeatingQC = ['Ex', 'Gd', 'TA', 'Fa', 'Po']
Electrical = ['SBrkr', 'FuseA', 'FuseF', 'FuseP', 'Mix']
KitchenQual = ['Ex', 'Gd', 'TA', 'Fa', 'Po']
Functional = ['Typ', 'Min1', 'Min2', 'Mod', 'Maj1', 'Maj2', 'Sev', 'Sal']
FireplaceQu = ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA']
GarageFinish = ['Fin', 'RFn', 'Unf', 'NA']
GarageQual = ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA']
GarageCond = ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA']
PavedDrive = ['Y', 'P', 'NA']
PoolQC = ['Ex', 'Gd', 'TA', 'Fa', 'NA']
Fence = ['GdPrv', 'MnPrv', 'GdWo', 'MnWw', 'NA']

categories = [LotShape, LandSlope, Utilities, ExterQual, ExterCond, BsmtQual, BsmtCond, BsmtExposure, 
              BsmtFinType1, BsmtFinType2, HeatingQC, Electrical, KitchenQual, Functional, FireplaceQu, 
              GarageFinish, GarageQual, GarageCond, PavedDrive, PoolQC, Fence]

ply = ['BsmtFinSF1', 'TotalBsmtSF', '1stFlrSF', 'GrLivArea', 'GarageArea']

polynomial_features = make_pipeline(
    SimpleImputer(strategy="constant", fill_value=0, add_indicator=True), 
    RobustScaler(),
    PolynomialFeatures(degree = 2, include_bias = False, interaction_only = False)
)
continuous_processor = make_pipeline(
    SimpleImputer(strategy="constant", fill_value=0, add_indicator=True), 
    RobustScaler(),
    PowerTransformer(method='yeo-johnson')
    
)

discrete_processor = make_pipeline(
    SimpleImputer(strategy="constant", fill_value=0, add_indicator=True), 
    RobustScaler()
)

ordinal_processor = make_pipeline(
    SimpleImputer(strategy='constant', fill_value='NA'),
    OrdinalEncoder(
        categories=categories,
        handle_unknown="use_encoded_value",
        unknown_value=-2,
        encoded_missing_value=-1
        ),
    MinMaxScaler(feature_range=(0, 1)),
)

nominal_processor = make_pipeline(
    SimpleImputer(strategy='constant', fill_value='No'),
    OneHotEncoder(
        handle_unknown='infrequent_if_exist',
        drop='first'
    ),
)

preprocessor = make_column_transformer(
    (polynomial_features, ply),
    (continuous_processor, continuous), 
    (discrete_processor, discrete),
    (ordinal_processor, ordinal),
    (nominal_processor, nominal)
)


#l2_regulatization = sp.loguniform.rvs(1e-5,100, size=100) 
l1_regulatization = [0.1, 0.4, 0.5, 0.55, 0.6, 0.7, 0.9, 0.95, 0.99, 1]

inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)

lr_pipeline = make_pipeline(
    preprocessor,
    SelectFromModel(LassoCV()),
    TransformedTargetRegressor(
        regressor=LinearRegression(), 
        func=np.log10, 
        inverse_func=sp.special.exp10
    )
)

ridge_pipeline = make_pipeline(
    preprocessor,
    SelectFromModel(LassoCV()),
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
    SelectFromModel(LassoCV()),
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
    SelectFromModel(LassoCV()),
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
        ('lr', lr_pipeline), 
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