import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    OneHotEncoder,
    RobustScaler,
    OrdinalEncoder,
    LabelEncoder,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from scipy.stats import chi2_contingency, ttest_ind, zscore
import matplotlib.pyplot as plt
import seaborn as sns

# missing data
# outlier
# normalization and scaling
# feature selection and feature extraction
# Encoding categorical
# Handling imbalanced data

data = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
feature = data.columns
t_feature = test.columns
label = feature[-1]
t_label = t_feature[-1]
mcar_tag = "MCAR"
mar_tag = "MAR"
mnar_tag = "MNAR"

# for col in feature:
#     sns.histplot(data[col])
#     plt.title(f"{col} histogram")
#     plt.show()


def extract_datatype(df, name="Dataset"):
    numeric_data = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_data = df.select_dtypes(include=["object"]).columns.tolist()
    return numeric_data, categorical_data


def analyze(df, name="dataset"):
    missing_count = df.isnull().sum()
    missing_percent = (missing_count / len(df)) * 100
    missing_tsumary = pd.DataFrame(
        {
            "missing_count": missing_count,
            "missing_percent": missing_percent,
            "dataframe type": df.dtypes,
        }
    )
    # print(f"{name} : {missing_tsumary[missing_count > 0]}")


# analyze(data, "Train")
# analyze(test, "Test")
missing_count = data.isnull().sum()
missing_columns = missing_count[missing_count > 0].index.tolist()
t_missing_count = test.isnull().sum()
t_missing_columns = t_missing_count[t_missing_count > 0].index.tolist()
numerical_data, categorical_data = extract_datatype(data, "Train")
t_numerical_data, t_categorical_data = extract_datatype(test, "Test")


def numerical_impute(df, missing, numerical_data, name="Dataset"):
    missing_data = [col for col in numerical_data if col in missing]
    if len(missing_data) <= 0:
        return
    for col in missing_data:
        if df[col].nunique() < 10:
            impute = SimpleImputer(strategy="median")
            df[[col]] = impute.fit_transform(df[[col]])
        else:
            impute = SimpleImputer(strategy="mean")
            df[[col]] = impute.fit_transform(df[[col]])
    print(f"{name} : {df.isnull().sum().sort_values(ascending=False)}")


def categorical_impute(df, missing, categorical_data, name="Dataset"):
    missing_data = [col for col in categorical_data if col in missing]
    if len(missing_data) <= 0:
        return
    for col in missing_data:
        impute = SimpleImputer(strategy="most_frequent")
        df[[col]] = impute.fit_transform(df[[col]])
    print(f"{name} : {df.isnull().sum().sort_values(ascending=False)}")


numerical_impute(data, missing_columns, numerical_data, "Train")
numerical_impute(data, t_missing_columns, t_numerical_data, "Test")
categorical_impute(test, missing_columns, categorical_data, "Train")
categorical_impute(test, t_missing_columns, t_categorical_data, "Test")


data_imputed = data.copy()
test_imputed = test.copy()
if "Price_JPY" in data_imputed.columns:
    data_imputed = data_imputed.drop(columns=["Price_JPY"])
numerical_data, categorical_data = extract_datatype(data_imputed, "Train")
t_numerical_data, t_categorical_data = extract_datatype(test_imputed, "Test")


def outlier_analyze(df, data):
    Q1 = df[data].quantile(0.25)
    Q3 = df[data].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df = df[data].clip(lower, upper, axis=0)


outlier_analyze(data_imputed, numerical_data)
outlier_analyze(test_imputed, t_numerical_data)

scaler = RobustScaler()
data_imputed[numerical_data] = scaler.fit_transform(data_imputed[numerical_data])
test_scaler = [col for col in t_numerical_data if col != "ID"]
test_imputed[test_scaler] = scaler.transform(test_imputed[test_scaler])

categorical_encode = pd.concat(
    [data_imputed[categorical_data], test_imputed[t_categorical_data]]
).astype(str)
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
encoder.fit(categorical_encode)
if len(categorical_data) > 0:
    encoded = encoder.transform(data_imputed[categorical_data])
    encoded_df = pd.DataFrame(
        encoded, columns=encoder.get_feature_names_out(categorical_data)
    )
    data_imputed.drop(columns=categorical_data, inplace=True)
    data_imputed = pd.concat([data_imputed, encoded_df], axis=1)

if len(t_categorical_data) > 0:
    t_encoded = encoder.transform(test_imputed[t_categorical_data])
    t_encoded_df = pd.DataFrame(
        t_encoded, columns=encoder.get_feature_names_out(t_categorical_data)
    )
    test_imputed.drop(columns=t_categorical_data, inplace=True)
    test_imputed = pd.concat([test_imputed, t_encoded_df], axis=1)

feature = [col for col in data_imputed.columns if col != "Price_JPY"]
x = data_imputed[feature]
y = data["Price_JPY"]
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=18)
x_test = test_imputed[feature]

train_numerical, train_categorical = extract_datatype(x_train, "Train")
val_numerical, val_categorical = extract_datatype(x_val, "Val")

model = Ridge()
param_model = {"alpha": [0.01, 0.02, 0.03, 0.1, 0.2, 0.3, 1, 2, 3, 10, 100]}
grid = GridSearchCV(model, param_model, cv=10, scoring="neg_mean_squared_error")
grid.fit(x_train, y_train)

print(f"Best alpha : {grid.best_params_['alpha']}")
print(f"Best score : {-grid.best_score_} and {grid.best_score_}")
model = grid.best_estimator_

feature_importance = pd.DataFrame(
    {
        "Feature": feature,
        "Coefficient": model.coef_,
        "Abs_Coefficient": np.abs(model.coef_),
    }
).sort_values("Abs_Coefficient", ascending=False)
print(f"\n⭐ TOP 5 FEATURES QUAN TRỌNG NHẤT:")
for i, (_, row) in enumerate(feature_importance.head(5).iterrows(), 1):
    print(f"   {i}. {row['Feature']}: {row['Coefficient']:,.2f}")


def calculate_metrics(y_true, y_pred, dataset_name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"\n📈 {dataset_name.upper()} PERFORMANCE:")
    print(f"   🎯 RMSE: {rmse:,.0f} JPY")
    print(f"   🎯 MAE: {mae:,.0f} JPY")
    print(f"   🎯 R² Score: {r2:.4f}")

    if r2 >= 0.8:
        performance = "Xuất sắc"
    elif r2 >= 0.6:
        performance = "Tốt"
    elif r2 >= 0.4:
        performance = "Trung bình"
    else:
        performance = "Cần cải thiện"

    print(f"   💬 Đánh giá: {performance}")

    return rmse, mae, r2


y_train_pred = model.predict(x_train)
y_val_pred = model.predict(x_val)

train_rmse, train_mae, train_r2 = calculate_metrics(y_train, y_train_pred, "Train")
val_rmse, val_mae, val_r2 = calculate_metrics(y_val, y_val_pred, "Val")

test_pred = model.predict(x_test)
submission = pd.DataFrame({"ID": test_imputed["ID"], "Price_JPY": test_pred})
submission_filename = "house_price_submission.csv"
submission.to_csv(submission_filename, index=False)
