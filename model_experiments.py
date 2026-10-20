
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

# Generate synthetic data (not company data)
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

results = []

# Automatically test every nonempty feature combination
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

        results.append({
            "features": ", ".join(combo),
            "n_features": len(combo),
            "r2": r2_score(y_test, predictions),
            "rmse": np.sqrt(
                mean_squared_error(y_test, predictions)
            ),
        })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    "rmse", ascending=True
)

print("\nTop 10 model combinations:\n")
print(results_df.head(10).to_string(index=False))

results_df.to_csv("experiment_results.csv", index=False)
