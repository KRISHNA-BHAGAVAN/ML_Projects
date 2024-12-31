import cv2
import pandas as pd

# Load student details from Excel
student_data = pd.read_excel("C:/Users/acer/Documents/student_details.xlsx")

# Create a dictionary where each Rollno maps to a tuple of (Student_Name, Student_Type, Section)
student_details = dict(zip(student_data['Rollno'], zip(student_data['Student_Name'], student_data['student_type'], student_data['section'])))

def display_dynamic_name(frame, roll_number, left, top):
    # Retrieve name, type, and section from the Excel data
    name, student_type, section = student_details.get(roll_number, ("Unknown", "Unknown", "Unknown"))
    
    # Display the name, type, and section near the detected face
    cv2.putText(frame, f"Name: {name}", (left-30, top - 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f"Student Type: {student_type}", (left-30, top - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f"Section: {section}", (left-30, top - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# Usage in main file:
# When a match is found, call this function to display the student's details
# display_dynamic_name(frame, matched_name, left, top)
