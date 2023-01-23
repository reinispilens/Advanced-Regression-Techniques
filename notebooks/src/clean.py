
import pandas as pd

def clean(df):
        #0.01
    outliers = [88, 185, 375, 523, 533, 635, 636, 691, 705, 769, 825, 1173, 1182,
            1298, 1337] 
        #0.02
    outliers = [  39,   87,   88,  178,  185,  197,  250,  291,  335,  375,  440,
             520,  523,  533,  614,  635,  636,  649,  664,  691,  705,  738,
             747,  769,  778,  803,  825,  828,  898,  914,  921,  942, 1011,
            1061, 1173, 1181, 1182, 1230, 1234, 1268, 1298, 1326, 1337, 1349] 
        #0.03
    outliers = [  39,   48,   87,   88,  125,  178,  185,  197,  198,  250,  291,
             307,  335,  349,  375,  386,  431,  434,  440,  496,  515,  520,
             523,  533,  581,  614,  635,  636,  649,  664,  691,  705,  738,
             747,  769,  778,  798,  803,  825,  828,  843,  897,  898,  914,
             921,  942,  954,  977, 1011, 1030, 1061, 1142, 1169, 1173, 1181,
            1182, 1219, 1228, 1230, 1234, 1243, 1268, 1283, 1298, 1323, 1326,
            1337, 1349, 1373, 1386, 1387, 1423, 1449]

    train_df = df.drop(outliers)
    
    dfx = train_df.copy()
    label = dfx.pop("SalePrice")
    
    return dfx, label
