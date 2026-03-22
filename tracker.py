import cv2
import mediapipe as mp
import time

last_trigger_time = 0
cooldown_period = 5

# Create hand detector
mp_hands = mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Drawing utility
mp_draw = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    current_time = time.time()# begins the cooldown timer
    status = "No Gesture Detected"
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = mp_hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame, handLms, 
                mp.solutions.hands.HAND_CONNECTIONS
            )

            # landamark titles ->
            # Wrist
            wrist = handLms.landmark[0]

            # Thumb
            thumb_cmc = handLms.landmark[1]
            thumb_mcp = handLms.landmark[2]
            thumb_ip  = handLms.landmark[3]
            thumb_tip = handLms.landmark[4]

            # Index Finger
            index_mcp = handLms.landmark[5]
            index_pip = handLms.landmark[6]
            index_dip = handLms.landmark[7]
            index_tip = handLms.landmark[8]

            # Middle Finger
            middle_mcp = handLms.landmark[9]
            middle_pip = handLms.landmark[10]
            middle_dip = handLms.landmark[11]
            middle_tip = handLms.landmark[12]

            # Ring Finger
            ring_mcp = handLms.landmark[13]
            ring_pip = handLms.landmark[14]
            ring_dip = handLms.landmark[15]
            ring_tip = handLms.landmark[16]

            # Pinky
            pinky_mcp = handLms.landmark[17]
            pinky_pip = handLms.landmark[18]
            pinky_dip = handLms.landmark[19]
            pinky_tip = handLms.landmark[20]

            if thumb_tip.y < thumb_mcp.y:
                status = "Thumbs Up"
                cv2.putText(frame, 'Thumbs Up!', (10, 70), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
            else:
                status = "Thumbs Down"
                cv2.putText(frame, 'Thumbs Down!', (10, 70), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

            if index_tip.y < index_mcp.y and middle_tip.y < middle_mcp.y and ring_tip.y < ring_mcp.y and pinky_tip.y < pinky_mcp.y:
                status = "open hand"
            elif index_tip.y > index_mcp.y and middle_tip.y > middle_mcp.y and ring_tip.y > ring_mcp.y and pinky_tip.y > pinky_mcp.y:
                status = "closed fist"
            elif index_pip.y < ring_tip.y and middle_pip.y < pinky_tip.y:
                status = "peace sign"
            elif pinky_pip.y < ring_dip.y and pinky_pip.y < middle_dip.y and thumb_tip.y < index_tip.y:
                status = "right on!"
            
            
    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_PLAIN, 1, (225, 225, 0), 2)
    cv2.imshow("Hand Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if status == "close":
        if current_time - last_trigger_time > cooldown_period:
            print("closing")
            last_trigger_time = current_time
            break

cap.release()
cv2.destroyAllWindows()