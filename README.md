# Project: Environmental Impact of Military Activities
# Author: Samarth Singh

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier

# -------------------------------
# 1. Sample Dataset
# -------------------------------

data = {
    "activity_intensity": [2, 4, 6, 8, 10],
    "air_quality": [80, 70, 60, 50, 40],
    "water_quality": [75, 65, 55, 45, 35],
    "soil_health": [85, 75, 65, 55, 45],
    "biodiversity": [90, 80, 70, 60, 50],
    "risk_level": ["Low", "Low", "Moderate", "High", "High"]
}

df = pd.DataFrame(data)

# -------------------------------
# 2. Regression Model
# Predict Air Quality based on Activity
# -------------------------------

X_reg = df[["activity_intensity"]]
y_reg = df["air_quality"]

reg_model = LinearRegression()
reg_model.fit(X_reg, y_reg)

predicted_air = reg_model.predict([[7]])

print("Predicted Air Quality for activity level 7:", predicted_air[0])

# -------------------------------
# 3. Classification Model
# Predict Risk Level
# -------------------------------

X_clf = df[["activity_intensity", "air_quality", "water_quality", "soil_health", "biodiversity"]]
y_clf = df["risk_level"]

clf_model = DecisionTreeClassifier()
clf_model.fit(X_clf, y_clf)

test_data = [[7, 55, 50, 60, 65]]
predicted_risk = clf_model.predict(test_data)

print("Predicted Risk Level:", predicted_risk[0])

# -------------------------------
# 4. Rule-Based Risk Function
# -------------------------------

def calculate_risk(score):
    if score >= 6:
        return "High"
    elif score >= 3:
        return "Moderate"
    else:
        return "Low"

risk_score = 7
print("Rule-Based Risk:", calculate_risk(risk_score))

# -------------------------------
# 5. Pollution Sources
# -------------------------------

pollution_sources = [
    "Fuel Leaks",
    "Explosives",
    "Toxic Chemicals",
    "Vehicle Emissions"
]

print("\nMajor Pollution Sources:")
for source in pollution_sources:
    print("-", source)

# -------------------------------
# 6. Mitigation Strategies
# -------------------------------

solutions = [
    "Use Cleaner Fuels",
    "Restrict Training Zones",
    "AI-based Monitoring Systems",
    "Proper Waste Disposal"
]

print("\nMitigation Strategies:")
for sol in solutions:
    print("-", sol)
