# src/evaluate.py
import pickle
import joblib
from sklearn.metrics import accuracy_score

# Load data
with open("data/X_test.pkl", "rb") as f:
    X_test = pickle.load(f)
with open("data/y_test.pkl", "rb") as f:
    y_test = pickle.load(f)

# Load models
clf = joblib.load("data/mlp_primary.pkl")
clf2 = joblib.load("data/mlp_secondary.pkl")
clf3 = joblib.load("data/logreg_misclf.pkl")

# Predictions
pred_primary = clf.predict(X_test)
pred_secondary = clf2.predict(X_test)
misclf_pred = clf3.predict(X_test)

# Combine predictions
final_pred = []
for i in range(len(y_test)):
    if misclf_pred[i] == 1:
        final_pred.append(pred_secondary[i])
    else:
        final_pred.append(pred_primary[i])

# Accuracy
acc_primary = accuracy_score(y_test, pred_primary)
acc_final = accuracy_score(y_test, final_pred)
print(f"Primary MLP Accuracy: {acc_primary:.4f}")
print(f"Final Combined Accuracy: {acc_final:.4f}")
