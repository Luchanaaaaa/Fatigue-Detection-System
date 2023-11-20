import sys
import dlib
import cv2
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QHBoxLayout
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
import numpy as np
from PyQt5.QtCore import QUrl
from facetracking import LandmarkProcessor







class VideoWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fatigue Detection")
        # Video label
        self.video_label = QLabel()

        # Status label
        self.status_label = QLabel()
        self.status_label.setFixedSize(50, 50)
        self.status_label.setStyleSheet("background-color: grey")

        # Layout
        self.layout = QHBoxLayout()
        self.layout.addWidget(self.video_label)
        self.layout.addWidget(self.status_label)
        self.setLayout(self.layout)

        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise ValueError("Could not open webcam.")
        ret, frame = self.cap.read()
        if not ret:
            self.cap.release()
            raise IOError("Webcam can not be accessed.")

        # Timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        # Dlib
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

        # EAR parameters
        self.EAR_THRESHOLD_ACTIVE = 0.28
        self.EAR_THRESHOLD_FATIGUE = 0.27
        self.EAR_THRESHOLD_SLEEP = 0.17
        self.EAR_CONSEC_FRAMES = 60     # TODO: 60 frames = 2 seconds
        self.EAR_frame_counter = 0

        # MAR parameters
        self.MAR_THRESHOLD_YAWN = 0.5  # TODO:  MAR threshold for yawning
        self.MAR_CONSEC_FRAMES = 60  # TODO： Number of consecutive frames the MAR must be below the threshold
        self.MAR_frame_counter = 0

        # Alarm player
        self.player = QMediaPlayer()
        self.player.setMedia(QMediaContent(QUrl.fromLocalFile("alert.mp3")))


    def update_frame(self):
        # Capture frame-by-frame
        ret, frame = self.cap.read()
        if ret:
            # Detect faces in the image
            faces = self.detector(frame, 1)
            # Draw a rectangle around the face and display 68 facial landmarks
            for face in faces:
                x, y, w, h = (face.left(), face.top(), face.width(), face.height())
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                landmarks = self.predictor(frame, face)
                processor = LandmarkProcessor(landmarks)
                smooth_ear = processor.calculate_ear()
                smooth_mar = processor.calculate_mouth()

                # Update the status label based on the average EAR and MAR
                print(smooth_ear, "\n")
                if smooth_ear >= self.EAR_THRESHOLD_ACTIVE:
                    self.status_label.setStyleSheet("background-color: green")  # Set status to green for active state
                elif self.EAR_THRESHOLD_SLEEP < smooth_ear < self.EAR_THRESHOLD_FATIGUE and smooth_mar >= self.MAR_THRESHOLD_YAWN:
                    self.EAR_frame_counter += 1
                    self.status_label.setStyleSheet(
                        "background-color: yellow")  # Set status to yellow for fatigue state
                    if self.EAR_frame_counter >= self.EAR_CONSEC_FRAMES or self.MAR_frame_counter >= self.MAR_CONSEC_FRAMES:
                        self.player.play()  # Play warning sound
                        QMessageBox.warning(self, "Warning", "Fatigue or Yawning detected! Please take a rest!")
                        self.EAR_frame_counter = 0  # Reset the EAR counter
                        self.MAR_frame_counter = 0  # Reset the MAR counter
                elif smooth_ear <= self.EAR_THRESHOLD_SLEEP:
                    self.status_label.setStyleSheet("background-color: red")  # Set status to red for sleep state
                    self.player.play()
                    QMessageBox.critical(self, "Critical Warning", "You might have fallen asleep!")
                    self.EAR_frame_counter = 0  # Reset the EAR counter


                for n in range(0, 68):
                    # Draw a point on each facial landmark
                    x = landmarks.part(n).x
                    y = landmarks.part(n).y
                    cv2.circle(frame, (x, y), 1, (255, 0, 0), -1)


            # Convert the processed frame back to QImage to display it in the GUI
            qimg = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888).rgbSwapped()
            self.video_label.setPixmap(QPixmap.fromImage(qimg))


    def closeEvent(self, event):
        self.cap.release()
        super().closeEvent(event)


# 程序入口点
if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = VideoWindow()
    win.show()
    sys.exit(app.exec_())
