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


def check_dependence_for_missing_col(check_data, missing_col, label):
    result = dict()

    check_data[f"{missing_col}_missing"] = check_data[missing_col].isnull().astype(int)

    for col in check_data.columns:
        if col == missing_col or col.endswith("_missing"):
            continue
        try:
            if check_data[col].dtype == "object" or check_data[col].nunique() < 10:
                contigency_table = pd.crosstab(
                    check_data[f"{missing_col}_missing"], check_data[col]
                )
                chi2, p, dof, expected = chi2_contingency(contigency_table)
            else:
                group_missing = check_data[check_data[f"{missing_col}_missing"] == 1][
                    label
                ].dropna()
                group_n_missing = check_data[check_data[f"{missing_col}_missing"] == 0][
                    label
                ].dropna()
                t_stat, p = ttest_ind(group_missing, group_n_missing, equal_var=False)
            result[col] = p
        except:
            continue
    return result


def check_tag(missing_columns, data):
    tag = dict()
    for col in missing_columns:
        check_data = data.copy()
        result = check_dependence_for_missing_col(check_data, col, label)
        dependence_total = len([val for val in result.values() if val < 0.05])
        if dependence_total / len(result) < 0.10:
            tag[col] = mcar_tag
        else:
            tag[col] = mar_mnar_tag
    return tag


def missing_handle(copy_data, tag):
    for col in missing_columns:
        missing_group = copy_data[col].isnull()
        n_missing_group = ~missing_group
        feature_cols = []
        for c in copy_data.columns:
            if c != col and copy_data[c].isnull().sum() == 0:
                feature_cols.append(c)
        if len(feature_cols) == 0:
            print(len(feature_cols))
            continue
        X_train = copy_data.loc[n_missing_group, feature_cols]
        Y_train = copy_data.loc[n_missing_group, col]
        X_pred = copy_data.loc[missing_group, feature_cols]
        if tag[col] == mcar_tag:
            if copy_data[col].dtype == "object":
                copy_data[col] = copy_data[col].fillna(copy_data.mode()[col][0])
            elif copy_data[col].dtype in ["int64"] and copy_data[col].nunique() < 10:
                copy_data[col] = copy_data[col].fillna(copy_data[col].median()[col])
            else:
                copy_data[col] = copy_data[col].fillna(copy_data[col].mean()[col])
        else:
            model = RandomForestRegressor()
            model.fit(X_train, Y_train)
            copy_data.loc[missing_group, col] = model.predict(X_pred)
        if copy_data[col].isnull().sum() > 0:
            print(f"Column {col} still has missing values after imputation.")
            copy_data[col] = copy_data[col].fillna(copy_data[col].mean())
    return copy_data


def check_outliner(copy_data):
    outliner = dict()
    for col in copy_data.columns:
        Q1 = copy_data[col].quantile(0.25)
        Q3 = copy_data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        higher = Q3 + 1.5 * IQR
        outliner[col] = len(
            copy_data[(copy_data[col] < lower) | (copy_data[col] > higher)]
        )
    return outliner


def encoder_handle(copy_data, label):
    """Handle categorical features"""
    copy_data = copy_data.copy()
    categorical_cols = copy_data.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = copy_data.select_dtypes(exclude=["object"]).columns.tolist()

    # Label encoding for the target variable if it's categorical
    if copy_data[label].dtype == "object":
        label_encoder = LabelEncoder()
        copy_data[label] = label_encoder.fit_transform(copy_data[label])

    return copy_data


def create_features(df):
    """Feature engineering"""
    df = df.copy()
    current_year = 2026

    # Age-related features
    if "YearBuilt" in df.columns:
        df["BuildingAge"] = current_year - df["YearBuilt"]
        df["IsNewBuilding"] = (df["BuildingAge"] <= 10).astype(int)

    # Area-related features
    if "LandArea_sqm" in df.columns and "TotalFloorArea_sqm" in df.columns:
        df["LandToFloorRatio"] = df["LandArea_sqm"] / (df["TotalFloorArea_sqm"] + 1)
        df["TotalUsableArea"] = df["TotalFloorArea_sqm"] + df["LandArea_sqm"]

    if "TotalFloorArea_sqm" in df.columns and "RoomCount" in df.columns:
        df["AreaPerRoom"] = df["TotalFloorArea_sqm"] / (df["RoomCount"] + 1)

    # Room-related features
    if all(col in df.columns for col in ["BedroomCount", "RoomCount"]):
        df["BedroomRatio"] = df["BedroomCount"] / (df["RoomCount"] + 1)

    if all(col in df.columns for col in ["BathroomCount", "RoomCount"]):
        df["BathroomRatio"] = df["BathroomCount"] / (df["RoomCount"] + 1)

    # Quality features
    if all(col in df.columns for col in ["ExteriorCondition", "InteriorCondition"]):
        df["OverallQuality"] = (df["ExteriorCondition"] + df["InteriorCondition"]) / 2
        df["IsHighQuality"] = (df["OverallQuality"] >= 4).astype(int)

    return df


def apply_transformations(df):
    """Áp dụng các biến đổi toán học"""
    df = df.copy()

    # Log transformations for skewed features với safe handling
    if "TotalFloorArea_sqm" in df.columns:
        # Kiểm tra và xử lý giá trị âm/NaN
        df["TotalFloorArea_sqm"] = df["TotalFloorArea_sqm"].fillna(0)
        df["TotalFloorArea_sqm"] = np.maximum(
            df["TotalFloorArea_sqm"], 0
        )  # Đảm bảo >= 0
        df["LogTotalFloorArea"] = np.log1p(df["TotalFloorArea_sqm"])

    if "LandArea_sqm" in df.columns:
        # Kiểm tra và xử lý giá trị âm/NaN
        df["LandArea_sqm"] = df["LandArea_sqm"].fillna(0)
        df["LandArea_sqm"] = np.maximum(df["LandArea_sqm"], 0)  # Đảm bảo >= 0
        df["LogLandArea"] = np.log1p(df["LandArea_sqm"])

    if "BuildingAge" in df.columns:
        # Kiểm tra và xử lý giá trị âm/NaN
        df["BuildingAge"] = df["BuildingAge"].fillna(0)
        df["BuildingAge"] = np.maximum(df["BuildingAge"], 0)  # Đảm bảo >= 0
        df["SqrtBuildingAge"] = np.sqrt(df["BuildingAge"])

    return df


# read data
data = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
feature = data.columns
t_feature = test.columns
label = feature[-1]
t_label = t_feature[-1]
mcar_tag = "MCAR"
mar_mnar_tag = "MAR/MNAR"
# analyze missing data
missing_analyst = data.isnull().sum().sort_values(ascending=False)
missing_data_filtered = missing_analyst[missing_analyst > 0]
missing_columns = missing_data_filtered.index.tolist()
missing_data = data[missing_columns]

# analyze missing test data
t_missing_analyst = test.isnull().sum().sort_values(ascending=False)
t_missing_data_filtered = t_missing_analyst[t_missing_analyst > 0]
t_missing_columns = t_missing_data_filtered.index.tolist()
t_missing_data = test[t_missing_columns]

copy_data = data.copy()
t_copy_data = test.copy()

tag = check_tag(missing_columns, copy_data)
t_tag = check_tag(t_missing_columns, t_copy_data)

# handle data
data_handle = missing_handle(copy_data, tag)
data_scaled = data_handle.copy()
data_scaled = data_scaled.drop(columns=label)

# handle test
t_data_handle = missing_handle(t_copy_data, t_tag)
t_data_scaled = t_data_handle.copy()
t_data_scaled = t_data_scaled.drop(columns=t_label)

# outliner data
outliner_total = check_outliner(data_scaled)
for key, value in outliner_total.items():
    if value > 0 and key in data_scaled.columns:
        scaler = RobustScaler()
        data_scaled[key] = scaler.fit_transform(data_scaled[[key]])
data_scaled[label] = data[label]

# outliner test
t_outliner_total = check_outliner(t_data_scaled)
for key, value in t_outliner_total.items():
    if value > 0 and key in t_data_scaled.columns:
        scaler = RobustScaler()
        t_data_scaled[key] = scaler.fit_transform(t_data_scaled[[key]])
t_data_scaled[t_label] = test[t_label]

encoder_data = encoder_handle(data_scaled, label)
t_encoder_data = encoder_handle(t_data_scaled, t_label)

# feature engineering data and test
process_data = create_features(encoder_data)
process_test = create_features(t_encoder_data)

# apply transform for data and test
process_data = apply_transformations(process_data)
process_test = apply_transformations(process_test)


def main():
    x = process_data.drop(columns=label)
    y = process_data[label]
    x_test = process_test.drop(columns=t_label)

    test_ids = range(len(x_test))

    common_col = list(set(x.columns) & set(x_test.columns))
    x = x[common_col]
    x_test = x_test[common_col]

    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=0.2, random_state=18
    )
    linear_reg = LinearRegression()
    linear_reg.fit(x_train, y_train)
    linear_val_predict = linear_reg.predict(x_val)
    linear_val_rmse = np.sqrt(mean_squared_error(y_val, linear_val_predict))
    linear_val_r2 = r2_score(y_val, linear_val_predict)

    print(f"Linear Regression (no regularization):")
    print(f"  Val RMSE: {linear_val_rmse:,.0f} JPY")
    print(f"  Val R²:   {linear_val_r2:.4f}")

    linear_test_pred = linear_reg.predict(x_test)
    submission = pd.DataFrame({"ID": test_ids, "Price_JPY": linear_test_pred})
    submission.to_csv("linear_submission.csv", index=False)
    print("\nSubmission saved to 'submission.csv'")

    linear_reg.fit(x, y)
    cv_scores = cross_val_score(
        linear_reg, x, y, cv=5, scoring="neg_mean_squared_error"
    )

    cv_rmse = np.sqrt(-cv_scores)
    print(f"5-Fold CV RMSE: {cv_rmse.mean():,.0f} (+/- {cv_rmse.std() * 2:,.0f})")

    print("making predict")
    test_pred = linear_reg.predict(x_test)

    submission = pd.DataFrame({"ID": test_ids, "Price_JPY": test_pred})

    submission.to_csv("submission.csv", index=False)
    print("\nSubmission saved to 'submission.csv'")


if __name__ == "__main__":
    main()
