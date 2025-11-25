import pickle
import glob


pickle_files = glob.glob("data_*.pickle")

all_data = []
all_labels = []

print(f"Found {len(pickle_files)} pickle files to merge...\n")

for file in pickle_files:
    print(f"Loading {file} ...")
    with open(file, "rb") as f:
        d = pickle.load(f)
        all_data.extend(d['data'])
        all_labels.extend(d['labels'])


with open("data_merged.pickle", "wb") as f:
    pickle.dump({'data': all_data, 'labels': all_labels}, f)

print(f"\nDone! Merged file created successfully: 'data_merged.pickle'")
print(f"Total samples: {len(all_data)}")
