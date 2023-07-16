import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_curve, roc_auc_score
from sklearn.metrics import auc, roc_curve, f1_score, accuracy_score, recall_score, precision_score
import joblib
import numpy as np
import pandas as pd
import pickle

import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')

# to visualize all the columns
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)



cat_col_final = ['NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'NAME_EDUCATION_TYPE']
num_col_final = ['TARGET', 'REGION_POPULATION_RELATIVE','EXT_SOURCE_3', 'DAYS_LAST_PHONE_CHANGE']

# built up the prprocess pipeline to avoid data leakage

num_feat = ['REGION_POPULATION_RELATIVE']
num_feat_median = ['EXT_SOURCE_3']
num_feat_mode = ['DAYS_LAST_PHONE_CHANGE']

cat_feat_onehot = ['NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR']
cat_feat_ordinal = ['NAME_EDUCATION_TYPE']

num_transformer = Pipeline(steps=[("scaler", StandardScaler())])


num_transformer_median = Pipeline(
    steps=[
        ("imputer_num", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

num_transformer_mode = Pipeline(
    steps=[("imputer_num", SimpleImputer(strategy="most_frequent")),
           ("scaler", StandardScaler())]
)

cat_transformer_onehot = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ("scaler", StandardScaler(with_mean=False))
    ]
)

cat_transformer_ordinal = Pipeline(
    steps=[
        ("ordinal", OrdinalEncoder(categories=[['Higher education', 'Secondary / secondary special', 'Incomplete higher',
 'Lower secondary', 'Academic degree']])),
        ("scaler", StandardScaler())
    ]
)
# construct the transformer objects above into a preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_transformer, num_feat),
        ("num_median", num_transformer_median, num_feat_median),
        ("num_mode", num_transformer_mode, num_feat_mode),
        ("cat_onehot", cat_transformer_onehot, cat_feat_onehot),
        ("cat_ordinal", cat_transformer_ordinal, cat_feat_ordinal)
    ]
)

df = pd.read_csv('application_train.csv')


y = df['TARGET']
X = df[cat_col_final + num_col_final].drop('TARGET', axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.2,
                                                    stratify = y,
                                                    random_state = 42
                                                   )

# Get the best model
steps=[("pre",preprocessor),
       ("lr", LogisticRegression(class_weight = 'balanced', penalty='l1', solver='liblinear', C=0.1))]

lg_opt = Pipeline(steps)
model_lgopt = lg_opt.fit(X_train, y_train)


pickle.dump(model_lgopt, open('model.pkl','wb'))