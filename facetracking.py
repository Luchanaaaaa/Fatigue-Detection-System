import numpy as np


class LandmarkProcessor:
    def __init__(self, landmarks, history_length=10):
        self.landmarks = landmarks
        self.HISTORY_LENGTH = history_length
        self.EAR_history = []
        self.MAR_history = []

    def eye_aspect_ratio(self, eye_points):
        eye = np.array([(self.landmarks.part(n).x, self.landmarks.part(n).y) for n in eye_points])
        vertical_1 = np.linalg.norm(eye[1] - eye[5])
        vertical_2 = np.linalg.norm(eye[2] - eye[4])
        horizontal = np.linalg.norm(eye[0] - eye[3])
        return (vertical_1 + vertical_2) / (2.0 * horizontal)

    def mouth_aspect_ratio(self):
        # Assuming these are the indexes for the mouth landmarks in the 68 point model
        top_lip_indexes = list(range(50, 53)) + list(range(61, 64))
        bottom_lip_indexes = list(range(56, 59)) + list(range(66, 68))

        top_lip = np.mean([(self.landmarks.part(n).x, self.landmarks.part(n).y) for n in top_lip_indexes], axis=0)
        bottom_lip = np.mean([(self.landmarks.part(n).x, self.landmarks.part(n).y) for n in bottom_lip_indexes], axis=0)

        lip_distance = np.linalg.norm(top_lip - bottom_lip)
        return lip_distance

    def calculate_ear(self):
        left_eye_points = range(36, 42)
        right_eye_points = range(42, 48)
        leftEAR = self.eye_aspect_ratio(left_eye_points)
        rightEAR = self.eye_aspect_ratio(right_eye_points)
        aveEAR = (leftEAR + rightEAR) / 2.0
        self.EAR_history.append(aveEAR)

        # Simply filter the EAR value to make it more smooth
        if len(self.EAR_history) > self.HISTORY_LENGTH:
            self.EAR_history.pop(0)

        smooth_ear = sum(self.EAR_history) / len(self.EAR_history)
        return smooth_ear

    def calculate_mouth(self):
        mar = self.mouth_aspect_ratio()
        self.MAR_history.append(mar)

        # Simply filter the MAR value to make it more smooth
        if len(self.MAR_history) > self.HISTORY_LENGTH:
            self.MAR_history.pop(0)

        smooth_mar = sum(self.MAR_history) / len(self.MAR_history)
        return smooth_mar

    def calculate_perclos(self):
        # Calculate the percentage of eye closure over the last frames
        if len(self.EAR_history) >= self.HISTORY_LENGTH:
            closed_frames = sum(1 for ear in self.EAR_history[-self.HISTORY_LENGTH:] if ear < self.EAR_THRESHOLD)
            perclos = (closed_frames / self.HISTORY_LENGTH) * 100
            return perclos
        else:
            return 0.0
    def get_drowsiness_status(perclos, threshold):
        if perclos < threshold:
            return "Non-Drowsy"
        else:
            return "Drowsy"
