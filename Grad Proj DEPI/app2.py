import flask
import base64
import pickle
import time
from collections import deque, Counter

import cv2
import numpy as np
import mediapipe as mp
from flask import Flask, request, render_template
from flask_socketio import SocketIO, emit

SMOOTHING_ALPHA = 0.5      
ADD_LETTER_DELAY = 3.0  
MAX_QUEUE_LEN = 20         
PREDICTION_THRESHOLD = 15  


app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'

socketio = SocketIO(app, cors_allowed_origins="*", logger=False, engineio_logger=False) 


MODEL_PATH = 'model_rfFINAAAAAL.p'
try:
    with open(MODEL_PATH, 'rb') as f:
        model_en = pickle.load(f)['model']
except FileNotFoundError:
    print(f"!!! Error: Model file '{MODEL_PATH}' not found !!!")
    exit()

english_letters = [chr(65 + i) for i in range(26)]
labels_dict_en = {i: english_letters[i] for i in range(len(english_letters))}
labels_dict_en[len(english_letters)] = "Space"

mp_hands = mp.solutions.hands
mp_hands_obj = mp_hands.Hands(
    static_image_mode=False, 
    max_num_hands=1, 
    min_detection_confidence=0.7, 
    min_tracking_confidence=0.7   
)

client_states = {}

def get_or_create_state(sid):
    if sid not in client_states:
        client_states[sid] = {
            'queue': deque(maxlen=MAX_QUEUE_LEN),
            'sentence': "",
            'last_char': "",
            'last_time': 0,
            'smooth_landmarks': None,
        }
    return client_states[sid]

@app.route('/')
def index():
    return render_template('index2.html')

@socketio.on('connect')
def handle_connect():
    sid = request.sid
    print(f"[CONNECT] {sid}")
    get_or_create_state(sid)

@socketio.on('disconnect')
def handle_disconnect():
    sid = request.sid
    print(f"[DISCONNECT] {sid}")
    client_states.pop(sid, None)

@socketio.on('frame')
def handle_frame(data):
    sid = request.sid
    state = get_or_create_state(sid)
    current_time = time.time()
    
    try:
        
        img_b64 = data.get('image', '').split(',', 1)[1]
        img_bytes = base64.b64decode(img_b64)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if frame is None:
            return

        H, W, _ = frame.shape
        
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False 
        results = mp_hands_obj.process(frame_rgb)
        frame_rgb.flags.writeable = True

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            current_landmark_list = hand_landmarks.landmark
            
            
            if state['smooth_landmarks'] is None:
                state['smooth_landmarks'] = [{'x': l.x, 'y': l.y} for l in current_landmark_list]
            else:
                for i in range(len(current_landmark_list)):
                    state['smooth_landmarks'][i]['x'] = SMOOTHING_ALPHA * current_landmark_list[i].x + (1 - SMOOTHING_ALPHA) * state['smooth_landmarks'][i]['x']
                    state['smooth_landmarks'][i]['y'] = SMOOTHING_ALPHA * current_landmark_list[i].y + (1 - SMOOTHING_ALPHA) * state['smooth_landmarks'][i]['y']
            
            
            x_, y_ = [], []
            data_aux = []
            
            for landmark in state['smooth_landmarks']:
                x_.append(landmark['x'])
                y_.append(landmark['y'])
                
            min_x, min_y = min(x_), min(y_)
            
            for landmark in state['smooth_landmarks']:
                data_aux.append(landmark['x'] - min_x)
                data_aux.append(landmark['y'] - min_y)

            
            prediction = model_en.predict([np.asarray(data_aux)])
            predicted_character = labels_dict_en[int(prediction[0])]
            state['queue'].append(predicted_character)
            
            
            most_common_char, count = Counter(state['queue']).most_common(1)[0]
            
            stable_char = ""
            if count > PREDICTION_THRESHOLD: 
                stable_char = most_common_char
                if (most_common_char != state['last_char'] or current_time - state['last_time'] > ADD_LETTER_DELAY):
                    if most_common_char == "Space":
                        state['sentence'] += " "
                    else:
                        state['sentence'] += most_common_char
                    
                    state['last_char'] = most_common_char
                    state['last_time'] = current_time
                    state['queue'].clear()
                    
                    emit('sentence_update', {'sentence': state['sentence']})

            
            x1, y1 = int(min(x_) * W) - 20, int(min(y_) * H) - 20
            x2, y2 = int(max(x_) * W) + 20, int(max(y_) * H) + 20
            bounding_box = [x1, y1, x2, y2]

            
            emit('hand_data', {
                'landmarks': state['smooth_landmarks'], 
                'box': bounding_box,                   
                'char': stable_char if stable_char else most_common_char
            })
                
        else:
            state['smooth_landmarks'] = None
            state['queue'].clear()
            emit('hand_data', {'landmarks': [], 'box': [], 'char': ''})

    except Exception as e:
        print(f"[ERROR] {sid}: {e}")

@socketio.on('clear_sentence')
def handle_clear():
    sid = request.sid
    state = get_or_create_state(sid)
    state['sentence'] = ""
    state['last_char'] = ""
    state['queue'].clear()
    emit('sentence_update', {'sentence': ""})

@socketio.on('backspace')
def handle_backspace():
    sid = request.sid
    state = get_or_create_state(sid)
    if state['sentence']:
        state['sentence'] = state['sentence'][:-1]
        state['last_char'] = "" 
        state['queue'].clear()
        emit('sentence_update', {'sentence': state['sentence']})

if __name__ == '__main__':
    print('Starting Flask Server...')
    socketio.run(app, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)