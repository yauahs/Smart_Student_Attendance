import cv2
import os
import numpy as np

def capture_and_save_image(image_folder, image_name, image_index):
    # Open the default camera (camera index 0)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    try:
        # Capture a single frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Couldn't capture frame.")
            return

        # Create the images folder if it doesn't exist
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)

        # Save the image in the images folder with the specified name and index
        image_path = os.path.join(image_folder, f"{image_name}_{image_index}.jpg")
        cv2.imwrite(image_path, frame)

        print(f"Image captured and saved to '{image_path}'.")

    finally:
        # Release the camera
        cap.release()

if __name__ == "__main__":
    # User input for image name and index
    image_name = input("Enter the Name of Student :: ")
    image_index = input("Enter the Roll Number of Student :: ")

    # Folder to save captured images
    image_folder = "Faces"

    capture_and_save_image(image_folder, image_name, image_index)
