import os
import pickle
from tqdm import tqdm  
import mediapipe as mp
import cv2


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'

data = []
labels = []


dirs = os.listdir(DATA_DIR)
print(f"Starting preprocessing for {len(dirs)} classes...\n")


for dir_ in tqdm(dirs, desc="Processing classes"):
    images = os.listdir(os.path.join(DATA_DIR, dir_))
    
    for img_path in tqdm(images, desc=f"Class {dir_}", leave=False):
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        if img is None:
            print(f"Skipping unreadable image: {img_path}")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

       
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                data_aux = []
                x_ = []
                y_ = []

                for lm in hand_landmarks.landmark:
                    x_.append(lm.x)
                    y_.append(lm.y)

                for lm in hand_landmarks.landmark:
                    data_aux.append(lm.x - min(x_))
                    data_aux.append(lm.y - min(y_))

                
                if len(data_aux) == 42:
                    data.append(data_aux)
                    labels.append(dir_)
                else:
                    print(f"Skipping image in '{dir_}' (invalid landmark count: {len(data_aux)})")


with open('data_mahmoud.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print(f"\nDone! Data saved successfully to 'data.pickle'.")
print(f"Total valid samples: {len(data)}")
