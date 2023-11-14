import cv2
import dlib
from scipy.spatial import distance

def calculate_ear(eye_landmarks):
    # Calculate the Euclidean distances between the vertical eye landmarks
    # (p2-p6, p3-p5)
    left_eye = distance.euclidean(eye_landmarks[1], eye_landmarks[5])
    right_eye = distance.euclidean(eye_landmarks[2], eye_landmarks[4])

    # Calculate the Euclidean distance between the horizontal eye landmarks
    # (p1-p4)
    horizontal = distance.euclidean(eye_landmarks[0], eye_landmarks[3])

    # Calculate the Eye Aspect Ratio (EAR)
    ear = (left_eye + right_eye) / (2.0 * horizontal)
    return ear

def get_drowsiness_status(perclos):
    if perclos < 0.075:
        return "Awake"
    elif 0.075 <= perclos < 0.15:
        return "Questionable"
    else:
        return "Drowsy"

# Load the face and eye detector models from dlib
face_detector = dlib.get_frontal_face_detector()
eye_detector = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize variables for eye closure tracking
total_frames = 0
closed_frames = 0
eye_threshold = 0.25  # Adjust this threshold as needed

# Initialize the webcam or video capture
cap = cv2.VideoCapture(0)  # 0 for the default camera, or specify a video file

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_detector(gray)

    for face in faces:
        landmarks = eye_detector(gray, face)
        
        # Extract eye landmarks
        left_eye_landmarks = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]
        right_eye_landmarks = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]

        # Calculate EAR for each eye
        left_ear = calculate_ear(left_eye_landmarks)
        right_ear = calculate_ear(right_eye_landmarks)

        # Check if eyes are closed
        if left_ear < eye_threshold and right_ear < eye_threshold:
            closed_frames += 1

        total_frames += 1

        # Visualize eye landmarks on the frame
        for (x, y) in left_eye_landmarks + right_eye_landmarks:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

    # Calculate PERCLOS
    perclos = (closed_frames / total_frames) * 100

    # Get drowsiness status based on PERCLOS
    drowsiness_status = get_drowsiness_status(perclos)

    # Display drowsiness status on the frame
    cv2.putText(frame, f'Drowsiness Status: {drowsiness_status}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display PERCLOS value on the frame
    cv2.putText(frame, f'PERCLOS: {perclos:.2f}%', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow("Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
