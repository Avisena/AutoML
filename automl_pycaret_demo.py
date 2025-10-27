# ============================================================
# AutoML Classification Demo with PyCaret
# Author: Vi 😎
# ============================================================

from pycaret.classification import *
from sklearn.datasets import load_wine
import pandas as pd

# === 1️⃣ Load dataset ===
print("🔹 Loading dataset...")
data = pd.read_csv("hr_train_augmented.csv")
data.rename(columns={"target": "Attrition"}, inplace=True)

print("✅ Dataset loaded. Sample:")
print(data.head(), "\n")

# === 2️⃣ Setup environment ===
print("🔹 Setting up PyCaret environment...")
clf = setup(
    data=data,
    target='Attrition',
    session_id=123,
    normalize=True,         # tetap bisa pakai ini
    log_experiment=False,   # matiin logging mlflow biar simple
    html=False              # matiin UI pop-up di console mode
)

print("✅ Setup complete.\n")

# === 3️⃣ Compare models (AutoML magic ✨) ===
print("🔹 Comparing models... (this might take a minute)")
best_model = compare_models()
print("✅ Best model found:")
print(best_model, "\n")

# === 4️⃣ Evaluate the best model ===
print("🔹 Evaluating model...")
evaluate_model(best_model)

# === 5️⃣ Make predictions ===
print("🔹 Making predictions on dataset...")
predictions = predict_model(best_model, data=data)
print("✅ Predictions done. Sample output:")
print(predictions.head(), "\n")

# === 6️⃣ Save the trained model ===
save_model(best_model, "best_iris_model")
print("💾 Model saved as 'best_iris_model.pkl'")

# === 7️⃣ Reload and re-predict ===
print("🔹 Loading saved model...")
loaded_model = load_model("best_iris_model")
print("✅ Model reloaded.")

new_preds = predict_model(loaded_model, data=data)
print("🔹 Sample predictions from reloaded model:")
print(new_preds.head())

print("\n🚀 Done! AutoML training complete. Enjoy your model 😎")
