import json
import re
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

CURRENT_YEAR = 2026
RANDOM_STATE = 42

# 1. LOAD
print("Loading data...")
df_train = pd.read_csv("train.csv")
df_train = df_train.drop(columns="id")

# 2. CLEANING: brand model mislabel (majority brand rule)
print("Cleaning brand-model mislabels...")
model_brand_counts = (df_train.groupby(["model", "brand"]).size().reset_index(name="count"))
majority_brand_map = (model_brand_counts.sort_values("count", ascending=False).drop_duplicates(subset="model", keep="first").set_index("model")["brand"])

df_train["majority_brand"] = df_train["model"].map(majority_brand_map)
mismatch_mask = df_train["brand"] != df_train["majority_brand"]
print(f"  dropped {mismatch_mask.sum()} mislabeled rows " f"({100*mismatch_mask.sum()/len(df_train):.2f}%)")
df_train = df_train[~mismatch_mask].drop(columns="majority_brand").reset_index(drop=True)

# car_age + target
df_train["car_age"] = CURRENT_YEAR - df_train["model_year"]
y_full = np.log1p(df_train["price"].copy())
X_full = df_train.copy()


def clean_features(df):
    df = df.copy()
    df["fuel_type"] = df["fuel_type"].replace(["\u2013", "not supported"], np.nan).fillna("Unknown")
    for col in ["ext_col", "int_col"]:
        df[col] = df[col].replace("\u2013", np.nan).fillna("Unknown").str.strip().str.title()
    weird_trans = ["2", "F", "SCHEDULED FOR OR IN PRODUCTION", "Transmission Overdrive Switch", "\u2013"]
    df["transmission"] = df["transmission"].replace(weird_trans, np.nan).fillna("Unknown")
    df["clean_title"] = df["clean_title"].fillna("Unknown").str.strip().str.title()
    df["accident"] = df["accident"].fillna("None reported").str.strip().str.title()
    return df


X_full = clean_features(X_full)

X_full["milage_per_year"] = X_full["milage"] / (X_full["car_age"] + 1)

# 4. OUTLIER REMOVAL: price ratio vs brand+model median
print("Removing price-placeholder outliers...")
brand_model_median = X_full.groupby(["brand", "model"])["price"].median()
X_full["group_median"] = X_full.set_index(["brand", "model"]).index.map(brand_model_median)
X_full["price_ratio"] = X_full["price"] / X_full["group_median"]
outlier_mask = (X_full["price_ratio"] > 5) | (X_full["price_ratio"] < 0.2)
print(f"  removed {outlier_mask.sum()} rows ({100*outlier_mask.sum()/len(X_full):.2f}%)")
X_full = X_full[~outlier_mask].copy()
y_full = y_full[~outlier_mask].copy()
X_full.drop(columns=["group_median", "price_ratio"], inplace=True)

# 5. ENGINE FEATURES
def engine_features(df):
    df = df.copy()
    df["horsepower"] = df["engine"].str.extract(r"(\d+\.?\d*)HP")[0].astype(float)
    df["engine_size"] = df["engine"].str.extract(r"(\d+\.\d+)L")[0].astype(float)
    cyl1 = df["engine"].str.extract(r"(\d+)\s*Cylinder")[0]
    cyl2 = df["engine"].str.extract(r"(?:I|V|W|H)(\d+)")[0]
    df["cylinder"] = cyl1.fillna(cyl2).astype(float)
    df["cylinder_missing"] = df["cylinder"].isna().astype(int)
    df["turbo"] = df["engine"].str.contains("Turbo", case=False, na=False).astype(int)
    df["hybrid"] = df["engine"].str.contains("Hybrid|Electric|Plug-In", case=False, na=False).astype(int)
    df["horsepower"] = df["horsepower"].fillna(df["horsepower"].median())
    df["engine_size"] = df["engine_size"].fillna(df["engine_size"].median())
    df["cylinder"] = df["cylinder"].fillna(df["cylinder"].median())
    df.drop(columns=["engine"], inplace=True)
    return df


hp_median = X_full["engine"].str.extract(r"(\d+\.?\d*)HP")[0].astype(float).median()
engsize_median = X_full["engine"].str.extract(r"(\d+\.\d+)L")[0].astype(float).median()
_cyl1 = X_full["engine"].str.extract(r"(\d+)\s*Cylinder")[0]
_cyl2 = X_full["engine"].str.extract(r"(?:I|V|W|H)(\d+)")[0]
cyl_median = _cyl1.fillna(_cyl2).astype(float).median()

X_full = engine_features(X_full)
X_full["hp_per_liter"] = X_full["horsepower"] / (X_full["engine_size"] + 1e-5)
X_full["hp_per_cylinder"] = X_full["horsepower"] / (X_full["cylinder"] + 1e-5)

# 6. TRANSMISSION FEATURES
def get_trans_type(x):
    if pd.isna(x):
        return "Unknown"
    x = str(x).lower()
    if "cvt" in x or "variable" in x:
        return "CVT"
    if "dual shift" in x or "dct" in x:
        return "DCT"
    if "manual" in x or "m/t" in x:
        return "Manual"
    if "automatic" in x or "a/t" in x:
        return "Automatic"
    return "Unknown"


def transmission_features(df):
    df = df.copy()
    df["transmission_type"] = df["transmission"].apply(get_trans_type)
    df["transmission_speed"] = df["transmission"].str.extract(
        r"(\d+)\s*[- ]?\s*speed", flags=re.IGNORECASE
    )[0]
    df["transmission_speed"] = pd.to_numeric(df["transmission_speed"], errors="coerce").fillna(0)
    df.drop(columns=["transmission"], inplace=True)
    return df


X_full = transmission_features(X_full)

# drop car_age (perfectly correlated with model_year)
X_full.drop(columns=["car_age"], inplace=True)

price_col = X_full["price"]
X_full = X_full.drop(columns=["price"]).copy()

# 7. SIMPAN SAMPLE UNTUK EDA TAB (Sebelum split/encoding)
eda_df = X_full.copy()
eda_df["price"] = price_col.values
eda_sample = eda_df.sample(n=min(30000, len(eda_df)), random_state=RANDOM_STATE)
eda_sample.to_csv("artifacts/eda_data.csv", index=False)
print(f"Saved EDA sample: {eda_sample.shape}")

# 8. SPLIT
X_train, X_val, y_train, y_val = train_test_split(X_full, y_full, test_size=0.2, random_state=RANDOM_STATE)
X_train = X_train.reset_index(drop=True)
X_val = X_val.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_val = y_val.reset_index(drop=True)

# 9. ENCODING
print("Encoding...")
# label encode accident & clean_title
le_accident = LabelEncoder().fit(X_train["accident"])
le_clean_title = LabelEncoder().fit(X_train["clean_title"])
X_train["accident"] = le_accident.transform(X_train["accident"])
X_val["accident"] = le_accident.transform(X_val["accident"])
X_train["clean_title"] = le_clean_title.transform(X_train["clean_title"])
X_val["clean_title"] = le_clean_title.transform(X_val["clean_title"])

# OHE fuel_type & transmission_type
ohe_cols = ["fuel_type", "transmission_type"]
ohe_encoder = OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore")
encoded_train = ohe_encoder.fit_transform(X_train[ohe_cols])
encoded_train_df = pd.DataFrame(encoded_train, columns=ohe_encoder.get_feature_names_out(ohe_cols), index=X_train.index)
X_train = pd.concat([X_train.drop(columns=ohe_cols), encoded_train_df], axis=1)

encoded_val = ohe_encoder.transform(X_val[ohe_cols])
encoded_val_df = pd.DataFrame(encoded_val, columns=ohe_encoder.get_feature_names_out(ohe_cols), index=X_val.index)
X_val = pd.concat([X_val.drop(columns=ohe_cols), encoded_val_df], axis=1)


# OOF target encoding (brand, model)
def oof_target_encoding(X_tr, y_tr, X_v, cols, n_splits=5, smoothing=10, random_state=42):
    X_tr = X_tr.copy()
    X_v = X_v.copy()
    global_mean = y_tr.mean()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    full_maps = {}
    for col in cols:
        oof_encoded = np.zeros(len(X_tr))
        for tr_idx, val_idx in kf.split(X_tr):
            tmp_X = X_tr.iloc[tr_idx]
            tmp_y = y_tr.iloc[tr_idx]
            temp = pd.DataFrame({col: tmp_X[col], "target": tmp_y})
            stats = temp.groupby(col)["target"].agg(mean="mean", count="count")
            enc_map = (stats["count"] * stats["mean"] + smoothing * global_mean) / (stats["count"] + smoothing)
            oof_encoded[val_idx] = X_tr.iloc[val_idx][col].map(enc_map).fillna(global_mean)
        X_tr[f"{col}_te"] = oof_encoded

        temp_full = pd.DataFrame({col: X_tr[col], "target": y_tr})
        stats_full = temp_full.groupby(col)["target"].agg(mean="mean", count="count")
        enc_map_full = (stats_full["count"] * stats_full["mean"] + smoothing * global_mean) / (stats_full["count"] + smoothing)
        X_v[f"{col}_te"] = X_v[col].map(enc_map_full).fillna(global_mean)
        full_maps[col] = {"map": enc_map_full.to_dict(), "global_mean": float(global_mean), "smoothing": smoothing}
    return X_tr, X_v, full_maps


cols_target_encode = ["brand", "model"]
X_train, X_val, target_encode_maps = oof_target_encoding(X_train, y_train, X_val, cols=cols_target_encode, n_splits=5, smoothing=10, random_state=RANDOM_STATE)
X_train.drop(columns=["brand", "model"], inplace=True)
X_val.drop(columns=["brand", "model"], inplace=True)

# frequency encoding ext_col & int_col
ext_freq_map = X_train["ext_col"].map(X_train["ext_col"].value_counts()).to_frame()
ext_freq_map = X_train["ext_col"].value_counts().to_dict()
int_freq_map = X_train["int_col"].value_counts().to_dict()
X_train["ext_col_freq"] = X_train["ext_col"].map(ext_freq_map)
X_train["int_col_freq"] = X_train["int_col"].map(int_freq_map)
X_val["ext_col_freq"] = X_val["ext_col"].map(ext_freq_map).fillna(0)
X_val["int_col_freq"] = X_val["int_col"].map(int_freq_map).fillna(0)
X_train.drop(columns=["ext_col", "int_col"], inplace=True)
X_val.drop(columns=["ext_col", "int_col"], inplace=True)

feature_columns = X_train.columns.tolist()
print("Final feature columns:", feature_columns)

# 10. TRAIN FINAL MODEL — GridSearchCV persis seperti di notebook
print("Running GridSearchCV (XGBoost)...")
xgb_param_grid = {
    "n_estimators": [600, 800],
    "max_depth": [5, 7],
    "learning_rate": [0.03, 0.05],
    "subsample": [0.8],
    "colsample_bytree": [0.8],
}

xgb_grid = GridSearchCV(
    XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1, tree_method="hist"),
    xgb_param_grid,
    cv=KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
    scoring="r2",
    n_jobs=1,
    verbose=1,
)
xgb_grid.fit(X_train, y_train)

print("Best XGB params :", xgb_grid.best_params_)
print("Best CV R2      :", xgb_grid.best_score_)

# best_params untuk disimpan ke metadata (bukan hardcode)
best_params = xgb_grid.best_params_

# CV scores dari GridSearchCV
cv_r2 = xgb_grid.best_score_
# ambil RMSE log-scale dari fold terbaik
best_idx = xgb_grid.best_index_
cv_rmse_log = float(-xgb_grid.cv_results_["mean_test_score"].mean())  # placeholder; pakai CV R2 saja

# best estimator sudah di-fit oleh GridSearchCV;
# evaluate_model di notebook refits on full X_train — lakukan hal yang sama
model = xgb_grid.best_estimator_
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_val_pred   = model.predict(X_val)

r2_train  = r2_score(y_train, y_train_pred)
rmse_train = np.sqrt(mean_squared_error(np.expm1(y_train), np.expm1(y_train_pred)))
r2_val    = r2_score(y_val, y_val_pred)
rmse_val  = np.sqrt(mean_squared_error(np.expm1(y_val), np.expm1(y_val_pred)))
mae_val   = mean_absolute_error(np.expm1(y_val), np.expm1(y_val_pred))

print(f"\n===== XGB (Tuned) =====")
print(f"R2 Train  : {r2_train:.4f}  |  RMSE Train : {rmse_train:,.4f}")
print(f"R2 Val    : {r2_val:.4f}  |  RMSE Val   : {rmse_val:,.4f}")
print(f"MAE Val   : ${mae_val:,.0f}")

# 11. DROPDOWN / SLIDER METADATA
brand_model_map = (X_full.groupby("brand")["model"].apply(lambda s: sorted(s.unique().tolist())).to_dict())
brand_list = sorted(X_full["brand"].unique().tolist())

ext_col_options = X_full["ext_col"].value_counts().head(25).index.tolist()
int_col_options = X_full["int_col"].value_counts().head(25).index.tolist()
fuel_type_options = sorted(X_full["fuel_type"].unique().tolist())
transmission_type_options = sorted(X_full["transmission_type"].unique().tolist())
accident_options = sorted(X_full["accident"].unique().tolist())
clean_title_options = sorted(X_full["clean_title"].unique().tolist())
transmission_speed_options = sorted(X_full["transmission_speed"].unique().tolist())

metadata = {
    "feature_columns": feature_columns,
    "ohe_cols": ohe_cols,
    "current_year": CURRENT_YEAR,
    "metrics": {
        "r2_train": r2_train,
        "rmse_train": rmse_train,
        "r2_val": r2_val,
        "rmse_val": rmse_val,
        "mae_val": mae_val,
        "cv_r2": cv_r2,
        "cv_rmse_log": cv_rmse_log,
        "n_rows_used": int(len(X_full)),
        "n_rows_raw": 188533,
        "n_outlier_dropped": int(outlier_mask.sum()) + int(mismatch_mask.sum()),
    },
    "best_params": best_params,
    "medians": {
        "horsepower": float(hp_median),
        "engine_size": float(engsize_median),
        "cylinder": float(cyl_median),
    },
    "ranges": {
        "model_year": [int(X_full["model_year"].min()), int(X_full["model_year"].max())],
        "milage": [int(X_full["milage"].min()), int(X_full["milage"].max())],
        "horsepower": [float(X_full["horsepower"].min()), float(X_full["horsepower"].max())],
        "engine_size": [float(X_full["engine_size"].min()), float(X_full["engine_size"].max())],
        "cylinder": [float(X_full["cylinder"].min()), float(X_full["cylinder"].max())],
        "price": [float(price_col.min()), float(price_col.max())],
    },
    "options": {
        "brand_list": brand_list,
        "brand_model_map": brand_model_map,
        "ext_col_options": ext_col_options,
        "int_col_options": int_col_options,
        "fuel_type_options": fuel_type_options,
        "transmission_type_options": transmission_type_options,
        "accident_options": accident_options,
        "clean_title_options": clean_title_options,
        "transmission_speed_options": transmission_speed_options,
    },
}

with open("artifacts/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

preprocessing = {
    "le_accident": le_accident,
    "le_clean_title": le_clean_title,
    "ohe_encoder": ohe_encoder,
    "ohe_cols": ohe_cols,
    "target_encode_maps": target_encode_maps,
    "ext_freq_map": ext_freq_map,
    "int_freq_map": int_freq_map,
    "feature_columns": feature_columns,
}

joblib.dump(preprocessing, "artifacts/preprocessing.joblib")
joblib.dump(model, "artifacts/model.joblib")

# feature importance for the model performance tab
importances = pd.Series(model.feature_importances_, index=feature_columns).sort_values(ascending=False)
importances.to_csv("artifacts/feature_importance.csv", header=["importance"])

# small actual vs predicted sample untuk plotting — pakai y_val asli (Series, bukan float override)
val_plot_df = pd.DataFrame({
    "actual": np.expm1(y_val.values),
    "predicted": np.expm1(y_val_pred),
})
val_plot_sample = val_plot_df.sample(n=min(4000, len(val_plot_df)), random_state=RANDOM_STATE)
val_plot_sample.to_csv("artifacts/val_predictions.csv", index=False)

print("\nDONE. Semua artefak tersimpan di ./artifacts/")