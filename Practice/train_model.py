# ======================= Car Price Prediction (Optimized XGBoost) ==========================
# Author: Fawad Raza
# Goal: Use RMSE (not MSE), compute MAE & R2, apply robust XGBoost tuning,
#       proper target transform (log1p) and scaling, save artifacts.
# =========================================================================================

import os
import re
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.metrics import (r2_score, mean_absolute_error,
                             mean_squared_error, mean_absolute_percentage_error)
import category_encoders as ce
import xgboost as xgb

# ---------------------- 1. Load data ----------------------
DATA_PATH = "data/file.csv"  # update if needed
df = pd.read_csv(DATA_PATH)
print("Dataset Shape:", df.shape)
print(df.head(3))

# ---------------------- 2. Basic cleaning ----------------------
# Remove rows with zero or missing price
df = df[df['price'].notna() & (df['price'] > 0)].copy()
df['price_numeric'] = df['price'].astype(float)

# Trim very large price outliers beyond 99th percentile
price_upper = df['price_numeric'].quantile(0.99)
df = df[df['price_numeric'] <= price_upper].copy()
print("After trimming outliers:", df.shape)

# Ensure numeric conversions
df['mileage'] = pd.to_numeric(df['mileage'], errors='coerce')
df['engine_capacity'] = pd.to_numeric(df['engine_capacity'], errors='coerce')
# If vehicle_age exists and is numeric, keep; otherwise try infer from model/year
if 'vehicle_age' not in df.columns:
    df['vehicle_age'] = np.nan

# ---------------------- 3. Feature engineering ----------------------
# Clean title / description
df['desc_clean'] = df['title'].astype(str).apply(lambda x: re.sub(r'[^a-zA-Z0-9 ]', ' ', x.lower()).strip())

# Keyword flags
df['is_automatic'] = df['desc_clean'].str.contains(r'\bautomatic\b', regex=True, na=False).astype(int)
df['is_hybrid'] = df['desc_clean'].str.contains(r'\bhybrid\b', regex=True, na=False).astype(int)
df['is_4wd'] = df['desc_clean'].str.contains(r'\b(4wd|awd)\b', regex=True, na=False).astype(int)
df['is_imported'] = df['desc_clean'].str.contains(r'\b(imported|japan|japanese|used import)\b', regex=True, na=False).astype(int)

# Ages & derived features
df['car_age'] = pd.to_numeric(df.get('vehicle_age', np.nan), errors='coerce')
# If car_age is missing but model looks like year, try extracting year from 'model' column
if df['car_age'].isna().sum() > 0 and 'model' in df.columns:
    # if model contains a year (e.g., "2017"), extract it and convert to age
    def extract_year_to_age(val):
        try:
            s = str(val)
            m = re.search(r'(19|20)\d{2}', s)
            if m:
                year = int(m.group(0))
                return pd.Timestamp.now().year - year
        except:
            pass
        return np.nan
    extracted = df['model'].apply(extract_year_to_age)
    df.loc[df['car_age'].isna(), 'car_age'] = extracted[df['car_age'].isna()]

# fallback median fill later
df['mileage_numeric'] = df['mileage']
df['mileage_per_year'] = df['mileage_numeric'] / (df['car_age'].fillna(df['car_age'].median()) + 1)
df['engine_per_age'] = df['engine_capacity'] / (df['car_age'].fillna(df['car_age'].median()) + 1)

# Normalize some categorical text
df['make_norm'] = df['title'].astype(str).str.split().str[0].str.lower()
df['model_norm'] = df['model'].astype(str).str.lower()
df['city_norm'] = df['city'].astype(str).str.lower()

# ---------------------- 4. Target encoding for model ----------------------
# Use TargetEncoder for high-cardinality 'model_norm'
target_encoder = ce.TargetEncoder(cols=['model_norm'], smoothing=0.3)
# Fit on full df to create encoded column (we'll avoid leakage because we still split later; for safer approach use nested CV)
df['model_encoded'] = target_encoder.fit_transform(df['model_norm'], df['price_numeric'])

# ---------------------- 5. Feature lists ----------------------
numeric_features = [
    'mileage_numeric', 'engine_capacity', 'car_age',
    'mileage_per_year', 'engine_per_age',
    'is_automatic', 'is_hybrid', 'is_4wd', 'is_imported', 'model_encoded'
]
# Keep only features present in df
numeric_features = [c for c in numeric_features if c in df.columns]

categorical_features = [c for c in ['make_norm', 'transmission', 'fuel_type', 'city_norm'] if c in df.columns]

print("Numeric features:", numeric_features)
print("Categorical features:", categorical_features)

# ---------------------- 6. Target transform (log) ----------------------
df['price_log'] = np.log1p(df['price_numeric'])
y = df['price_log']
X = df[numeric_features + categorical_features].copy()

# ---------------------- 7. Train/test split ----------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
print("Train/Test split sizes:", X_train.shape, X_test.shape)

# ---------------------- 8. Preprocessor (impute + robust scale + ohe) ----------------------
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', RobustScaler())
])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
], remainder='drop')

# ---------------------- 9. XGBoost base regressor ----------------------
xgb_base = xgb.XGBRegressor(
    objective='reg:squarederror',
    random_state=42,
    verbosity=0,
    n_jobs=-1
)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', xgb_base)
])

# ---------------------- 10. Hyperparameter tuning (optimize RMSE) ----------------------
param_distributions = {
    'regressor__n_estimators': [500, 800, 1000],
    'regressor__learning_rate': [0.01, 0.03, 0.05],
    'regressor__max_depth': [7, 9, 11],
    'regressor__subsample': [0.7, 0.8, 0.9],
    'regressor__colsample_bytree': [0.7, 0.8, 0.9],
    'regressor__min_child_weight': [1, 2, 4],
    'regressor__gamma': [0, 0.1, 0.2],
    'regressor__reg_alpha': [0, 0.1, 0.3],
    'regressor__reg_lambda': [1, 1.5, 2.0]
}

search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_distributions,
    n_iter=25,
    scoring='neg_root_mean_squared_error',  # optimize RMSE
    cv=5,
    random_state=42,
    n_jobs=-1,
    verbose=2
)

print("\n🔍 Starting RandomizedSearchCV (optimizing RMSE)...")
search.fit(X_train, y_train)
print("✅ Best Params:", search.best_params_)
print("✅ Best CV (neg RMSE):", search.best_score_)

best_model = search.best_estimator_

# ---------------------- 11. Evaluate on test set (compute RMSE, MAE, R2) ----------------------
y_pred_log = best_model.predict(X_test)
# convert back to original scale
y_pred = np.expm1(y_pred_log)
y_true = np.expm1(y_test)

mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_true, y_pred)
mape = mean_absolute_percentage_error(y_true, y_pred)

print("\n📊 Model Performance on TEST set:")
print(f"R² Score: {r2:.4f}")
print(f"MAE: {mae:,.0f}")
print(f"RMSE: {rmse:,.0f}")
print(f"MAPE: {mape*100:.2f}%")

# ---------------------- 12. Cross-validated R2 (optionally) ----------------------
cv_r2 = cross_val_score(best_model, X, y, cv=5, scoring='r2', n_jobs=-1)
print(f"\nCross-Validated R² mean: {cv_r2.mean():.4f} (std: {cv_r2.std():.4f})")

# ---------------------- 13. Feature importance (robust extraction) ----------------------
# Build feature names: numeric followed by onehot feature names
def get_feature_names_from_column_transformer(ct, numeric_feats, categorical_feats, X_sample=None):
    feature_names = []
    # numeric names are just numeric_feats
    feature_names.extend(numeric_feats)
    # categorical -> onehot feature names
    # we need fitted transformer
    for name, trans, cols in ct.transformers_:
        if name == 'cat':
            # trans is a Pipeline with fitted OneHotEncoder at step 'onehot'
            # find the onehot encoder
            ohe = None
            if hasattr(trans, 'named_steps') and 'onehot' in trans.named_steps:
                ohe = trans.named_steps['onehot']
            elif hasattr(trans, 'steps') and len(trans.steps) > 0:
                # fallback
                for step_name, step in trans.steps:
                    if isinstance(step, OneHotEncoder):
                        ohe = step
                        break
            if ohe is None:
                # fallback, just append categorical column names
                feature_names.extend(categorical_feats)
            else:
                # get categories for each input feature
                try:
                    cats = ohe.get_feature_names_out(cols)
                    feature_names.extend(list(cats))
                except Exception:
                    # older sklearn: construct manually
                    for i, c in enumerate(cols):
                        categories = ohe.categories_[i]
                        feature_names.extend([f"{c}__{cat}" for cat in categories])
    return feature_names

preproc = best_model.named_steps['preprocessor']
feature_names = get_feature_names_from_column_transformer(preproc, numeric_features, categorical_features)

xgb_reg = best_model.named_steps['regressor']
importances = xgb_reg.feature_importances_

# align length safety check
if len(importances) == len(feature_names):
    imp_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
else:
    # if mismatch, show top feature indices
    imp_df = pd.DataFrame({
        'feature': [f"f{i}" for i in range(len(importances))],
        'importance': importances
    })

imp_df = imp_df.sort_values('importance', ascending=False).head(25)

plt.figure(figsize=(10, 8))
sns.barplot(x='importance', y='feature', data=imp_df)
plt.title('Top Feature Importances (XGBoost)')
plt.tight_layout()
os.makedirs("assets", exist_ok=True)
plt.savefig("assets/xgb_feature_importance.png")
print("Feature importance plot saved to assets/xgb_feature_importance.png")

# ---------------------- 14. Save artifacts ----------------------
os.makedirs("artifacts", exist_ok=True)
joblib.dump(best_model, "artifacts/car_price_xgb_model.pkl")
joblib.dump(target_encoder, "artifacts/model_target_encoder.pkl")
metrics = {'r2': r2, 'mae': mae, 'rmse': rmse, 'mape': mape, 'cv_r2_mean': cv_r2.mean() if 'cv_r2' in locals() else None}
joblib.dump(metrics, "artifacts/metrics.pkl")
print("\n💾 Saved model and artifacts to artifacts/")

print("\n✅ Training complete — model ready for inference (uses RMSE as primary tuning metric).")
