
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, RobustScaler, StandardScaler, MinMaxScaler, QuantileTransformer, PowerTransformer, PolynomialFeatures, StandardScaler

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
