import pickle
import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque, Counter


model_en = pickle.load(open('model_rfFINAAAAAL.p', 'rb'))['model']


english_letters = [chr(65+i) for i in range(26)]  # A-Z
labels_dict_en = {i: english_letters[i] for i in range(len(english_letters))}
labels_dict_en[len(english_letters)] = "Space"   # ONLY SPACE


cap = cv2.VideoCapture(0)


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


hands = mp_hands.Hands(
    static_image_mode=False,
    min_detection_confidence=0.7,  
    min_tracking_confidence=0.7    
)


SMOOTHING_ALPHA = 0.5  
smoothed_landmark_list = None


sentence_en = ""
predictions_queue = deque(maxlen=20)
last_added_char = ""
last_time_added = time.time()
ADD_LETTER_DELAY = 3.0


scan_start_time = 0
scan_duration = 0.6
scanning = False


while True:
    data_aux = []
    ret, frame = cap.read()
    if not ret:
        break
        
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    current_time = time.time()

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        

        current_landmark_list = hand_landmarks.landmark
        
        if smoothed_landmark_list is None:

            smoothed_landmark_list = [{'x': l.x, 'y': l.y} for l in current_landmark_list]
        else:
            for i in range(len(current_landmark_list)):
                smoothed_landmark_list[i]['x'] = SMOOTHING_ALPHA * current_landmark_list[i].x + (1 - SMOOTHING_ALPHA) * smoothed_landmark_list[i]['x']
                smoothed_landmark_list[i]['y'] = SMOOTHING_ALPHA * current_landmark_list[i].y + (1 - SMOOTHING_ALPHA) * smoothed_landmark_list[i]['y']
        

        

        mp_drawing.draw_landmarks(
             frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
           mp_drawing_styles.get_default_hand_landmarks_style(),
           mp_drawing_styles.get_default_hand_connections_style()
        )

        x_, y_ = [], []
        data_aux = []
        

        for landmark in smoothed_landmark_list:
            x_.append(landmark['x'])
            y_.append(landmark['y'])
            
        min_x, min_y = min(x_), min(y_)
        
        for landmark in smoothed_landmark_list:
            data_aux.append(landmark['x'] - min_x)
            data_aux.append(landmark['y'] - min_y)

        x1, y1 = int(min(x_) * W) - 20, int(min(y_) * H) - 20
        x2, y2 = int(max(x_) * W) + 20, int(max(y_) * H) + 20

        prediction = model_en.predict([np.asarray(data_aux)])
        predicted_character = labels_dict_en[int(prediction[0])]

        predictions_queue.append(predicted_character)
        most_common_char, count = Counter(predictions_queue).most_common(1)[0]

        if count > 15 and (most_common_char != last_added_char or current_time - last_time_added > ADD_LETTER_DELAY):
            if most_common_char == "Space":
                sentence_en += " "
            else:
                sentence_en += most_common_char

            last_added_char = most_common_char
            last_time_added = current_time
            scan_start_time = current_time
            scanning = True

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

        if scanning and current_time - scan_start_time < scan_duration:
            progress = (current_time - scan_start_time) / scan_duration
            scan_y = int(y1 + progress * (y2 - y1))
            cv2.line(frame, (x1, scan_y), (x2, scan_y), (0, 255, 0), 2)
        else:
            scanning = False

     
        cv2.putText(frame, most_common_char, (x1, y1 - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
    
    else:

        smoothed_landmark_list = None
        predictions_queue.clear() 


    if sentence_en.strip():
        text_size, _ = cv2.getTextSize(sentence_en, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)
        cv2.rectangle(frame, (20, 20), (20 + text_size[0] + 20, 20 + text_size[1] + 20), (255, 255, 255), -1)
        cv2.putText(frame, sentence_en, (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)

    cv2.putText(frame, "Language: EN", (30, H-30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    cv2.imshow("frame", frame)

    key = cv2.waitKey(1)

    if key == ord("q"):
        break

    elif key == ord("c"):  
        sentence_en = ""
        last_added_char = ""
        predictions_queue.clear()

    elif key == ord("z"): 
        if sentence_en:
            sentence_en = sentence_en[:-1]
        last_added_char = ""
        predictions_queue.clear()

cap.release()
cv2.destroyAllWindows()