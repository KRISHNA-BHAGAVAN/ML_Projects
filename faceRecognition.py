import cv2
import face_recognition
import pandas as pd
import requests
from io import BytesIO
import os
import pickle
from dynamic_display import display_dynamic_name

# Base URL
base_url = "https://info.aec.edu.in/AEC/StudentPhotos/"

# Load student details from Excel
file_path = "C:/Users/acer/Documents/student_details.xlsx"
df = pd.read_excel(file_path)
roll_numbers = df["Rollno"].tolist()

# Initialize lists for face encodings and names
known_face_encodings = []
known_face_names = []

# File paths to save encodings and names
encodings_file = "face_encodings.pkl"
names_file = "face_names.pkl"

# Check if encodings and names files exist
if os.path.exists(encodings_file) and os.path.exists(names_file):
    # Load encodings and names from files
    with open(encodings_file, 'rb') as f:
        known_face_encodings = pickle.load(f)
    with open(names_file, 'rb') as f:
        known_face_names = pickle.load(f)
    print("Loaded encodings and names from files.")
else:
    # Fetch and encode each image by roll number
    for roll_number in roll_numbers:
        url = f"{base_url}{roll_number}.jpg"
        response = requests.get(url)
        
        if response.status_code == 200:
            print(f"Successfully downloaded image for roll number {roll_number}.")
            # Load the image
            image = face_recognition.load_image_file(BytesIO(response.content))
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Encode the face
            encodings = face_recognition.face_encodings(image)
            
            if encodings:
                # Add encoding and corresponding name
                known_face_encodings.append(encodings[0])
                known_face_names.append(roll_number)
                print(f"Successfully encoded face for roll number {roll_number}.")
            else:
                print(f"Warning: No face detected for roll number {roll_number}.")
        else:
            print(f"Image not found for roll number {roll_number} (HTTP status {response.status_code}).")
    
    # Save encodings and names to files
    with open(encodings_file, 'wb') as f:
        pickle.dump(known_face_encodings, f)
    with open(names_file, 'wb') as f:
        pickle.dump(known_face_names, f)
    print("Saved encodings and names to files.")

# Start capturing video
video_capture = cv2.VideoCapture(0)

# Variables to store true and predicted labels
true_labels = []
predicted_labels = []

match_found = False
matched_image = None

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    if not ret:
        break

    # Find all face locations and encodings in the frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    best_match_distance = float('inf')  # Set a high initial distance
    best_match_index = -1

    # Loop through each face in the camera frame
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Check if the face matches any known faces
        for i, known_encoding in enumerate(known_face_encodings):
            # Calculate the distance (lower distance means a better match)
            distance = face_recognition.face_distance([known_encoding], face_encoding)[0]

            if distance < best_match_distance:  # Find the best match with the smallest distance
                best_match_distance = distance
                best_match_index = i

        # If a match is found with a low distance (high accuracy), draw a rectangle around the face
        if best_match_index != -1 and best_match_distance < 0.6:  # Threshold for a good match
            matched_name = known_face_names[best_match_index]
            matched_image = cv2.cvtColor(face_recognition.load_image_file(BytesIO(requests.get(f"{base_url}{matched_name}.jpg").content)), cv2.COLOR_RGB2BGR)
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame,"Match found", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            display_dynamic_name(frame, matched_name, left, top)
            match_found = True
            true_labels.append(matched_name)  # Append the true label
            predicted_labels.append(matched_name)  # Append the predicted label
            break  # Exit once a match is found
        else:
            true_labels.append("Unknown")
            predicted_labels.append("Unknown")

    # Show the video feed with the match (if found)
    cv2.imshow("Video Feed", frame)
    
    # Show the matched image if match found, else cycle through known images
    if match_found and matched_image is not None:
        cv2.imshow("Matched Image", matched_image)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
video_capture.release()
cv2.destroyAllWindows()

# Save true and predicted labels for evaluation
with open("true_labels.pkl", 'wb') as f:
    pickle.dump(true_labels, f)
with open("predicted_labels.pkl", 'wb') as f:
    pickle.dump(predicted_labels, f)
