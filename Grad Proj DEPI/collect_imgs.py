import os
import cv2

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 28
dataset_size = 166  

cap = cv2.VideoCapture(0)

for j in range(number_of_classes):
    class_path = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_path):
        os.makedirs(class_path)

    print(f'Collecting data for class {j}')

    
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, 'Ready? Press "Q" to start :)', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        elif cv2.waitKey(25) & 0xFF == 27:  # Esc
            cap.release()
            cv2.destroyAllWindows()
            exit()

    
    existing_files = [f for f in os.listdir(class_path) if f.endswith('.jpg') and f.split('.')[0].isdigit()]
    existing_numbers = [int(f.split('.')[0]) for f in existing_files]
    counter = max(existing_numbers) + 1 if existing_numbers else 0
    total_existing = len(existing_numbers)

   
    if total_existing >= dataset_size:
        print(f"Class {j} already has {total_existing} images. Skipping...")
        continue

    
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            break

        
        text = f'Class {j} | Image: {counter}/{dataset_size}  (Press ESC to stop)'
        cv2.putText(frame, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)

        key = cv2.waitKey(25) & 0xFF
        if key == 27:  
            print(f"Stopped collecting class {j} early at {counter} images.")
            break

        filename = os.path.join(class_path, f'{counter}.jpg')
        cv2.imwrite(filename, frame)
        counter += 1

    print(f"Finished collecting for class {j} ({counter} images total).")

cap.release()
cv2.destroyAllWindows()
