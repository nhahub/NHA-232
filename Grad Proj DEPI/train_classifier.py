import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

print("Loading data...")
data_dict = pickle.load(open('./data_merged.pickle', 'rb'))
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])
print(f"Loaded: {len(data)} samples")

x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels
)

print("\n🚀 Training RandomForest...")
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=42
)
model.fit(x_train, y_train)

print("\nEvaluating...")
y_pred = model.predict(x_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print(classification_report(y_test, y_pred))

with open("model_rfFINAAAAAL.p", "wb") as f:
    pickle.dump({'model': model}, f)

print("\nSaved as model_rf.p")