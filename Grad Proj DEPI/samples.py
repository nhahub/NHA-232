import pickle
from collections import Counter


with open("data_merged.pickle", "rb") as f:
    data_dict = pickle.load(f)

data = data_dict['data']
labels = data_dict['labels']


class_counts = Counter(labels)

print("\n📊 Dataset Summary:")
print("---------------------------")
for cls, count in class_counts.items():
    print(f"🖐️ Class '{cls}': {count} samples")

print("---------------------------")
print(f"Total samples: {len(data)}")
print(f"Unique classes: {len(class_counts)}")
