import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from mlxtend.regressor import StackingCVRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('./Middle_Dataset_final.csv')

# Convert the timestamp column to datetime to filter the required time period
df['timestamp'] = pd.to_datetime(df['timestamp'], format='%H:%M:%S').dt.time

# Filter the dataframe to include only rows between 6:00 AM and 10:00 PM
start_time = pd.to_datetime("06:00:00", format='%H:%M:%S').time()
end_time = pd.to_datetime("22:00:00", format='%H:%M:%S').time()

filtered_df = df[(df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)]

# Dropping the time and date columns for modeling purposes
filtered_df = filtered_df.drop(columns=['date', 'newdate', 'timestamp'])

# Define input features (X) and target variable (y)
X = filtered_df.drop(columns=['target_speed'])
y = filtered_df['target_speed']

# Define the column transformer
num_attribs = X.select_dtypes(include=[np.number]).columns.tolist()
cat_attribs = X.select_dtypes(include=['object']).columns.tolist()

num_pipeline = Pipeline([
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

full_pipeline = ColumnTransformer([
    ('num', num_pipeline, num_attribs),
    ('cat', cat_pipeline, cat_attribs)
])

# Apply the preprocessing pipeline to the data
X_preprocessed = full_pipeline.fit_transform(X)

# Split the preprocessed data into training and testing sets
X_train_preprocessed, X_test_preprocessed, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.33, random_state=42)

# Define a function to train models and evaluate performance
def algorithm_pipeline(X_train_data, X_test_data, y_train_data, y_test_data, model, param_grid, cv=10, scoring_fit='neg_mean_squared_error', scoring_test=r2_score, do_probabilities=False):
    gs = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, n_jobs=-1, scoring=scoring_fit, verbose=0)
    fitted_model = gs.fit(X_train_data, y_train_data)
    best_model = fitted_model.best_estimator_
    if do_probabilities:
      pred = fitted_model.predict_proba(X_test_data)
    else:
      pred = fitted_model.predict(X_test_data)
    score = scoring_test(y_test_data, pred)
    mse = mean_squared_error(y_test_data, pred)
    return [best_model, pred, score, mse]

# Define models and hyperparameters
models_to_train = [XGBRegressor(), CatBoostRegressor(logging_level='Silent'), GradientBoostingRegressor(), RandomForestRegressor()]
grid_parameters = [
    { # XGBoost
        'n_estimators': [400, 700, 1000],
        'colsample_bytree': [0.7, 0.8],
        'max_depth': [15, 20, 25],
        'reg_alpha': [1.1, 1.2, 1.3],
        'reg_lambda': [1.1, 1.2, 1.3],
        'subsample': [0.7, 0.8, 0.9]
    },
    { # CatBoost
        "iterations": [100, 200], 
        "learning_rate": [0.01, 0.1], 
        "depth": [4, 6]
    },
    { # Gradient Boosting
        "n_estimators": [100, 200, 500], 
        "learning_rate": [0.01, 0.1, 0.5], 
        "max_depth": [3, 5, 10]
    },
    { # Random Forest
        'max_depth': [5, 10, 15, 20], 
        'n_estimators': [100, 200, 400, 600, 900],
        'max_features': [2, 4, 6, 8, 10]
    }
]

# Train models and evaluate
models_preds_scores = []
for i, model in enumerate(models_to_train):
    params = grid_parameters[i]
    result = algorithm_pipeline(X_train_preprocessed, X_test_preprocessed, y_train, y_test, model, params, cv=10)
    models_preds_scores.append(result)

for result in models_preds_scores:
    print('Model: {0}, Score: {1}, MSE: {2}'.format(type(result[0]).__name__, result[2], result[3]))

# Define individual models for stacking
xgb = XGBRegressor()
catB = CatBoostRegressor(logging_level='Silent')
gradB = GradientBoostingRegressor()
rf = RandomForestRegressor()
decisionTree = DecisionTreeRegressor()
ridge = Ridge()
lasso = Lasso()
svr = SVR(kernel='linear')

# Implement stacking regressor
stack = StackingCVRegressor(regressors=(decisionTree, xgb, rf, gradB, catB),
                            meta_regressor=catB,
                            cv=10,
                            use_features_in_secondary=True,
                            store_train_meta_features=True,
                            shuffle=False,
                            random_state=42)

stack.fit(X_train_preprocessed, y_train)

# Predict and evaluate the stacking regressor
stack_pred = stack.predict(X_test_preprocessed)
stack_score = r2_score(y_test, stack_pred)
stack_mse = mean_squared_error(y_test, stack_pred)

print(f'Stacking Regressor R2 Score: {stack_score}')
print(f'Stacking Regressor Mean Squared Error: {stack_mse}')

# Visualization
# time_series = pd.date_range(start="06:00", end="22:00", freq="30T").strftime("%H:%M")

# plt.figure(figsize=(10, 6))
# plt.plot(time_series[:len(y_test)], y_test[:len(time_series)], label='Actual Speed', color='blue')
# plt.plot(time_series[:len(stack_pred)], stack_pred[:len(time_series)], label='Prediction Speed', color='yellow')
# plt.xlabel('Time')
# plt.ylabel('Speed')
# plt.title('Graph for speed and time')
# plt.xticks(rotation=45)
# plt.legend()
# plt.show()

# Create a time series for the plotting range
time_series = pd.date_range(start="06:00", end="22:00", freq="30T").strftime("%H:%M")

# Ensure that all arrays have the same length
min_length = min(len(time_series), len(y_test), len(stack_pred))
time_series = time_series[:min_length]
y_test = y_test.iloc[:min_length]
stack_pred = stack_pred[:min_length]

# Plot the actual vs. predicted speed
plt.figure(figsize=(10, 6))
plt.plot(time_series, y_test, label='Actual Speed', color='blue')
plt.plot(time_series, stack_pred, label='Predicted Speed', color='red')
plt.xlabel('Time')
plt.ylabel('Speed')
plt.title('Actual Speed vs Predicted Speed')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig('./image/Improved_Speed_Prediction.jpg')
plt.show()