import joblib

model = joblib.load("models/svm_klasifikasi.joblib")
print(model.feature_names_in_)
