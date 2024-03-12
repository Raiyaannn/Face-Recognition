from PIL import Image, ImageTk
import os 
import cv2
import face_recognition
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import simpledialog


class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition App")
        
        self.cap = cv2.VideoCapture(0)
        self.known_images, self.known_labels = self.load_known_faces_from_directory("C:/Users/Raiyan/Desktop/known_faces_directory")
        self.known_face_encodings = [face_recognition.face_encodings(img)[0] for img in self.known_images]
        
        self.create_widgets()

    def create_widgets(self):
        self.btn_register = tk.Button(self.root, text="Register New Face", command=self.register_face)
        self.btn_register.pack(pady=10)
        
        self.video_label = tk.Label(self.root)
        self.video_label.pack()

        self.quit_button = tk.Button(self.root, text="Quit", command=self.quit)
        self.quit_button.pack(pady=10)

    def register_face(self):
        name = simpledialog.askstring("Input", "Enter the name of the person:")
        if name:
            ret, frame = self.cap.read()
            if not ret:
                messagebox.showerror("Error", "Couldn't read frame from webcam")
                return

            face_locations = face_recognition.face_locations(frame)
            if len(face_locations) == 0:
                messagebox.showerror("Error", "No face detected. Please try again.")
                return
            elif len(face_locations) > 1:
                messagebox.showerror("Error", "Multiple faces detected. Please try again with only one face visible.")
                return

            face_encoding = face_recognition.face_encodings(frame, face_locations)[0]

            image_path = f"C:/Users/Raiyan/Desktop/known_faces_directory/{name}.jpg"
            cv2.imwrite(image_path, frame)

            self.known_images.append(frame)
            self.known_face_encodings.append(face_encoding)
            self.known_labels.append(name)

            messagebox.showinfo("Success", "Face registered successfully!")

    def load_known_faces_from_directory(self, directory_path):
        known_images = []
        known_labels = []
        for file_name in os.listdir(directory_path):
            if file_name.endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(directory_path, file_name)
                label = os.path.splitext(file_name)[0]
                known_images.append(face_recognition.load_image_file(image_path))
                known_labels.append(label)
        return known_images, known_labels

    def quit(self):
        self.cap.release()
        self.root.destroy()

    def start_recognition(self):
        ret, frame = self.cap.read()
        if not ret:
            messagebox.showerror("Error", "Couldn't read frame from webcam")
            return

        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            min_distance_index = np.argmin(distances)
            min_distance = distances[min_distance_index]

            if min_distance < 0.6:
                recognized_label = self.known_labels[min_distance_index]
            else:
                recognized_label = "Unknown"

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, recognized_label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)

        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        self.root.after(10, self.start_recognition)

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    app.start_recognition()
    root.mainloop()
