from pycaret.classification import load_model, predict_model
import pandas as pd

# === Load model yang udah disimpan ===
model = load_model("best_iris_model")
print("✅ Model loaded successfully!")

# === Contoh data baru yang mau diprediksi ===
# (Ganti nilai sesuai dataset lo)
new_data = pd.DataFrame({
    "sepal length (cm)": [5.1, 6.7, 4.9],
    "sepal width (cm)": [3.5, 3.0, 2.4],
    "petal length (cm)": [1.4, 5.2, 3.3],
    "petal width (cm)": [0.2, 2.3, 1.0],
})

print("\n🔹 New data:")
print(new_data)

# === Prediksi ===
predictions = predict_model(model, data=new_data)
print("\n🔹 Predictions:")
print(predictions)
