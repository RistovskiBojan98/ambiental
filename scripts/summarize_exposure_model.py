import pandas as pd
import numpy as np
import sys

from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ambiental.config import PROCESSED_DATA_PATH

df = pd.read_csv(PROCESSED_DATA_PATH)
df['Datetime'] = pd.to_datetime(df['Datetime'])
df.set_index('Datetime', inplace=True)
df.sort_index(inplace=True)

low = df['multi_pollutant_exposure'].quantile(0.25)
med = df['multi_pollutant_exposure'].quantile(0.50)
high = df['multi_pollutant_exposure'].quantile(0.75)
spike = df['contextual_pollution_score'].quantile(0.85)

threshold_multiplier = 0.85
adj_low = low * threshold_multiplier
adj_med = med * threshold_multiplier
adj_high = high * threshold_multiplier

print('Thresholds:')
print(f'  low (25%) = {low:.6f}')
print(f'  medium (50%) = {med:.6f}')
print(f'  high (75%) = {high:.6f}')
print(f'  spike (85% contextual) = {spike:.6f}')
print('')
print('Adjusted thresholds for sensitive profile:')
print(f'  adjusted low = {adj_low:.6f}')
print(f'  adjusted medium = {adj_med:.6f}')
print(f'  adjusted high = {adj_high:.6f}')

# Add exposure_trend if missing
if 'exposure_trend' not in df.columns:
    df['exposure_trend'] = df['multi_pollutant_exposure'].diff()

# build one-step prediction dataset
proc = df.copy()
proc['future_exposure'] = proc['multi_pollutant_exposure'].shift(-1)
features = [
    'CO(GT)',
    'NO2(GT)',
    'NOx(GT)',
    'C6H6(GT)',
    'pollution_score',
    'contextual_pollution_score',
    'T',
    'RH',
    'AH',
    'hour',
    'hour_weight',
    'exposure_trend',
]

proc = proc.dropna(subset=features + ['future_exposure']).copy()

X = proc[features]
y = proc['future_exposure']

split = int(len(proc) * 0.8)
X_train = X.iloc[:split]
X_test = X.iloc[split:]
y_train = y.iloc[:split]
y_test = y.iloc[split:]

# Linear regression baseline
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
mae_lr = mean_absolute_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)
print('Linear regression metrics:')
print(f'  RMSE = {rmse_lr:.6f}')
print(f'  MAE  = {mae_lr:.6f}')
print(f'  R^2  = {r2_lr:.6f}')
print('Coefficients:')
for feature, coeff in zip(features, lr.coef_):
    print(f'  {feature}: {coeff:.6f}')

# Random forest predictive model
rf = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=2,
    n_jobs=-1,
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
print('')
print('Random forest metrics:')
print(f'  RMSE = {rmse_rf:.6f}')
print(f'  MAE  = {mae_rf:.6f}')
print(f'  R^2  = {r2_rf:.6f}')
print('Feature importances:')
for feature, importance in sorted(zip(features, rf.feature_importances_), key=lambda x: -x[1]):
    print(f'  {feature}: {importance:.6f}')
