
import numpy as np
import pandas as pd
from itertools import combinations

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

from statsmodels.stats.outliers_influence import (
    variance_inflation_factor
)
from statsmodels.tools.tools import add_constant


# STEP 1: Generate synthetic data
rng = np.random.default_rng(42)
n = 300

data = pd.DataFrame({
    "price": rng.normal(10, 2, n),
    "advertising": rng.normal(50, 10, n),
    "promotions": rng.integers(0, 2, n),
    "temperature": rng.normal(70, 10, n),
    "competitor_price": rng.normal(12, 2, n),
})

data["sales"] = (
    1000
    - 30 * data["price"]
    + 8 * data["advertising"]
    + 120 * data["promotions"]
    + rng.normal(0, 50, n)
)

features = [
    "price",
    "advertising",
    "promotions",
    "temperature",
    "competitor_price",
]


# STEP 2: Function to calculate maximum VIF
def calculate_max_vif(X):
    if X.shape[1] == 1:
        return 1.0

    X_with_constant = add_constant(X, has_constant="add")

    vif_values = [
        variance_inflation_factor(
            X_with_constant.values, i
        )
        for i in range(1, X_with_constant.shape[1])
    ]

    return max(vif_values)


# STEP 3: Automatically test model combinations
results = []

for size in range(1, len(features) + 1):
    for combo in combinations(features, size):

        X = data[list(combo)]
        y = data["sales"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )

        model = LinearRegression()
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        r2 = r2_score(y_test, predictions)
        rmse = np.sqrt(
            mean_squared_error(y_test, predictions)
        )

        # Calculate multicollinearity using training data
        max_vif = calculate_max_vif(X_train)

        # Flag models with high VIF
        vif_status = (
            "High" if max_vif > 10
            else "Moderate" if max_vif > 5
            else "Low"
        )

        results.append({
            "features": ", ".join(combo),
            "n_features": len(combo),
            "r2": r2,
            "rmse": rmse,
            "max_vif": max_vif,
            "vif_status": vif_status,
        })


# STEP 4: Rank models by prediction error
results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "rmse", ascending=True
)

print("\nTop 10 model combinations:\n")
print(results_df.head(10).to_string(index=False))


# STEP 5: Identify models with acceptable VIF
stable_models = results_df[
    results_df["max_vif"] <= 5
]

print("\nTop 5 models with acceptable VIF:\n")
print(stable_models.head(5).to_string(index=False))


# STEP 6: Export experiment results
results_df.to_csv("experiment_results.csv", index=False)

print("\nExperiment results saved successfully.")
