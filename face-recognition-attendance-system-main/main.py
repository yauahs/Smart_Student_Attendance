import cv2
import numpy as np
import dlib
import csv
import os
import face_recognition
import pickle
import time
from csv import writer
import pandas as pd

# Import the HeadPoseEstimator class
from Pose_Estimation_Module import HeadPoseEstimator
# Initialize face and eye detectors
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Function to get face landmarks using dlib
def get_landmarks(im):
    rects = detector(im, 1)
    if len(rects) > 1:
        print("Too many faces")
        return np.matrix([0])
    if len(rects) == 0:
        print("Too few faces")
        return np.matrix([0])
    return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])

# Function to draw landmarks on the image
def place_landmarks(im, landmarks):
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(idx), pos, fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, fontScale=0.3, color=(0, 255, 255))
        cv2.circle(im, pos, 3, color=(0, 255, 255))
    return im

# Function to get upper lip landmarks
def upper_lip(landmarks):
    top_lip = []
    for i in range(50, 53):
        top_lip.append(landmarks[i])
    for j in range(61, 64):
        top_lip.append(landmarks[j])
    top_lip_point = np.squeeze(np.asarray(top_lip))
    top_mean = np.mean(top_lip_point, axis=0)
    return int(top_mean[1])

# Function to get lower lip landmarks
def low_lip(landmarks):
    lower_lip = []
    for i in range(65, 68):
        lower_lip.append(landmarks[i])
    for j in range(56, 59):
        lower_lip.append(landmarks[j])
    lower_lip_point = np.squeeze(np.asarray(lower_lip))
    lower_mean = np.mean(lower_lip_point, axis=0)
    return int(lower_mean[1])

# Function to decide whether a yawn has occurred based on lip distance
def decision(image):
    landmarks = get_landmarks(image)
    if landmarks.all() == [0]:
        return -10  # Dummy value to prevent error
    top_lip = upper_lip(landmarks)
    lower_lip = low_lip(landmarks)
    distance = abs(top_lip - lower_lip)
    return distance

# Function to find face encodings
def find_encodings(images_list):
    encode_list = []
    for img in images_list:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encode_list.append(encode)
    return encode_list

print("Loading face images...")
folder_path = 'Faces'
file_path = os.listdir(folder_path)

img_list = []
student_ids = []

for path in file_path:
    img_list.append(cv2.imread(os.path.join(folder_path, path)))
    student_ids.append(os.path.splitext(path)[0])

print("Finding face encodings...")
encode_list_know = find_encodings(img_list)
encode_list_know_with_ids = [encode_list_know, student_ids]

print("Saving face encodings...")
file = open('EncodingFile.p', 'wb')
pickle.dump(encode_list_know_with_ids, file)
file.close()

print("Loading face encodings...")
file = open('EncodingFile.p', 'rb')
encode_list_know_with_ids = pickle.load(file)
file.close()

encode_list_know, student_ids = encode_list_know_with_ids

path = "predictor/shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(path)
detector = dlib.get_frontal_face_detector()

head_pose_estimator = HeadPoseEstimator()

print("Loading attendance data...")
df = pd.read_csv('Attendance.csv')

# ... (previous code)

# Open or create the attendance.csv file for writing
csv_file_path = "Attendance.csv"

# Check if the file exists
file_exists = os.path.exists(csv_file_path)

# Define cap outside the loop
cap = cv2.VideoCapture(0)

# Create a folder to store frames
frames_folder = 'Frames'
os.makedirs(frames_folder, exist_ok=True)

with open(csv_file_path, 'a', newline='') as csvfile:
    fieldnames = ['Roll_No', 'Name', 'Date', 'Time', 'Yawn Count', 'Attentiveness', 'Eye Closed']
    csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    # Write the header only if the file is newly created
    if not file_exists:
        csv_writer.writeheader()

    yawns = 0
    eye_closed = 0

    while True:
        ret, frame = cap.read()
        if ret:
            landmarks = get_landmarks(frame)
            if landmarks.all() != [0]:
                l1 = []
                for k in range(48, 60):
                    l1.append(landmarks[k])
                l2 = np.asarray(l1)
                lips = cv2.convexHull(l2)
                cv2.drawContours(frame, [lips], -1, (0, 255, 0), 1)

                # Eye detection
                eyes = eye_cascade.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=5)
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)

                # Check if eyes are closed
                if len(eyes) > 0:
                    eye_closed = 1 
                else:
                    eye_closed = 0 

            distance = decision(frame)
            if distance > 30:
                yawns = 1
            else:
                yawns = 0

            # face recognition part
            faceCurFrame = face_recognition.face_locations(frame)
            encodeCurFrame = face_recognition.face_encodings(frame, faceCurFrame)

            attentiveness = "1"  # Assuming attentive by default

            for encodeFace, _ in zip(encodeCurFrame, faceCurFrame):
                matches = face_recognition.compare_faces(encode_list_know, encodeFace)
                faceDis = face_recognition.face_distance(encode_list_know, encodeFace)
                matcheIndex = np.argmin(faceDis)

                if matches[matcheIndex]:
                    name_csv = student_ids[matcheIndex]
                    roll_no_csv = student_ids[matcheIndex]

                    # Check attentiveness based on yawns and face recognition
                    if yawns == 1 or eye_closed == 1:  # Define your criteria for determining inattentiveness
                        attentiveness = "0"

                    # Write data to the CSV file
                    csv_writer.writerow({
                        'Roll_No': matcheIndex,  # Use the correct index
                        'Name': name_csv,
                        'Date': time.strftime("%d/%m/%Y"),
                        'Time': time.strftime("%H:%M:%S"),
                        'Yawn Count': yawns,
                        'Attentiveness': attentiveness,
                        'Eye Closed': eye_closed
                    })

                    # Flush the writer to ensure the data is written to the file
                    csvfile.flush()
            cv2.putText(frame, "Yawning: " + str(yawns), (50, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1.0, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)

            cv2.putText(frame, "Eyes Closed: " + str(eye_closed), (50, 150), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1.0, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)

            cv2.putText(frame, "Attentiveness: " + attentiveness, (50, 200), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1.0, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)

            cv2.imshow("Subject Yawn Count", frame)
            if cv2.waitKey(1) == 13:
                break
        else:
            continue

# Release video capture and close all windows
cap.release()
cv2.destroyAllWindows()
