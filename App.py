import kivy
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.uix.textinput import TextInput
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.core.window import Window
from kivy.graphics import Color, Rectangle, RoundedRectangle

import cv2
import numpy as np
import pickle
from mtcnn import MTCNN
from keras_facenet import FaceNet
from datetime import datetime, time
import csv
import os

# Set window size
Window.size = (1000, 700)
Window.clearcolor = (0.95, 0.95, 0.97, 1)

# BAI-4 Spring 2024 Timetable
TIMETABLE = {
    'MONDAY': [
        {'course': 'MTH262 - Statistics and Probability Theory', 'start': '08:30', 'end': '10:00', 'room': '208', 'instructor': 'Ms. Mudassar Bibi'},
        {'course': 'CSC270 - Database Systems', 'start': '10:00', 'end': '11:30', 'room': '208', 'instructor': 'Mr. Muhammad Harris (Add.)'},
    ],
    'TUESDAY': [
        {'course': 'CSC270 - Database Systems', 'start': '08:30', 'end': '10:00', 'room': '206', 'instructor': 'Mr. Muhammad Harris (Add.)'},
        {'course': 'MTH262 - Statistics and Probability Theory', 'start': '10:00', 'end': '11:30', 'room': '206', 'instructor': 'Ms. Mudassar Bibi'},
        {'course': 'CSC291 - Software Engineering', 'start': '11:30', 'end': '13:00', 'room': '206', 'instructor': 'Ms. Hina Jan'},
    ],
    'WEDNESDAY': [
        {'course': 'AIC354 - Machine Learning Fundamentals', 'start': '08:30', 'end': '11:30', 'room': 'CL-11', 'instructor': 'Mr. Umar Nauman'},
        {'course': 'CSC270 - Database Systems', 'start': '11:30', 'end': '14:30', 'room': 'CL-11', 'instructor': 'Mr. Muhammad Harris (Add.)'},
    ],
    'THURSDAY': [
        {'course': 'CSC270 - Database Systems', 'start': '10:30', 'end': '12:30', 'room': '211', 'instructor': 'Mr. Muhammad Harris (Add.)'},
    ],
    'FRIDAY': [
        {'course': 'AIC270 - Programming for Artificial Intelligence', 'start': '08:30', 'end': '10:30', 'room': 'CL-11 (Class)', 'instructor': 'Dr. Usman Yaseen'},
        {'course': 'AIC354 - Machine Learning Fundamentals', 'start': '10:30', 'end': '11:30', 'room': '211', 'instructor': 'Mr. Umar Nauman'},
        {'course': 'AIC270 - Programming for Artificial Intelligence', 'start': '11:30', 'end': '13:00', 'room': 'CL-11', 'instructor': 'Dr. Usman Yaseen'},
    ]
}

class AttendanceSystem(BoxLayout):
    def __init__(self, **kwargs):
        super(AttendanceSystem, self).__init__(**kwargs)
        self.orientation = 'vertical'
        self.padding = 15
        self.spacing = 10
        
        # Load models
        print("Loading models...")
        self.load_models()
        
        # Initialize detector and FaceNet
        self.detector = MTCNN()
        self.facenet = FaceNet()
        
        # Initialize front camera (usually index 1)
        self.capture = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        if not self.capture.isOpened():
            print("Front camera not found, using default camera.")
            self.capture = cv2.VideoCapture(0)
        
        # Attendance tracking
        self.attendance_marked = set()
        self.current_frame = None
        self.processing = False
        
        # Class information
        self.current_class_info = None
        
        # Student database with registration numbers
        self.student_database = {
            'ahmad': {'name': 'Ahmad Khan', 'reg_no': ''},
            'farhan': {'name': 'Farhan Khan', 'reg_no': ''}
        }
        
        # Create attendance CSV if not exists
        self.attendance_file = 'attendance_bai4_spring2024.csv'
        if not os.path.exists(self.attendance_file):
            with open(self.attendance_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Registration No', 'Name', 'Course', 'Room', 'Instructor', 'Date', 'Time', 'Class Start', 'Class End', 'Confidence', 'Status'])
        
        # UI Components
        self.create_ui()
        
        # Start camera update
        Clock.schedule_interval(self.update_camera, 1.0 / 30.0)
        
        # Update class info every second
        Clock.schedule_interval(self.update_current_class, 1.0)
        
        # Initialize current class
        self.update_current_class(0)
    
    def load_models(self):
        """Load the trained SVM model and label encoder"""
        try:
            with open('farhan_fcae.pkl', 'rb') as f:
                model_data = pickle.load(f)
                self.svm_model = model_data['model']
                self.encoder = model_data['encoder']
            print("Models loaded successfully!")
        except Exception as e:
            print(f"Error loading models: {e}")
            raise
    
    def get_current_class(self):
        """Get the current class based on real-time day and time"""
        now = datetime.now()
        day_name = now.strftime('%A').upper()
        current_time = now.time()
        
        if day_name not in TIMETABLE:
            return None
        
        for class_info in TIMETABLE[day_name]:
            start_time = datetime.strptime(class_info['start'], '%H:%M').time()
            end_time = datetime.strptime(class_info['end'], '%H:%M').time()
            
            # Check if current time is within class time (with 30 min buffer after class)
            buffer_end = datetime.combine(datetime.today(), end_time)
            buffer_end = (buffer_end + pd.Timedelta(minutes=30)).time() if 'pd' in dir() else end_time
            
            if start_time <= current_time <= end_time or (current_time >= start_time and current_time <= end_time):
                return class_info
        
        # If no exact match, find the next upcoming class today
        for class_info in TIMETABLE[day_name]:
            start_time = datetime.strptime(class_info['start'], '%H:%M').time()
            if current_time < start_time:
                return class_info
        
        return None
    
    def update_current_class(self, dt):
        """Update the current class information display"""
        class_info = self.get_current_class()
        
        if class_info:
            self.current_class_info = class_info
            self.course_label.text = f"📚 {class_info['course']}"
            self.time_label.text = f"🕐 {class_info['start']} - {class_info['end']}"
            self.room_label.text = f"🏫 Room: {class_info['room']}"
            self.instructor_label.text = f"👨‍🏫 {class_info['instructor']}"
            
            # Update current time display
            now = datetime.now()
            self.current_time_label.text = f"⏰ Current Time: {now.strftime('%I:%M:%S %p')}"
            
            # Check if class is active
            start_time = datetime.strptime(class_info['start'], '%H:%M').time()
            end_time = datetime.strptime(class_info['end'], '%H:%M').time()
            current_time = now.time()
            
            if start_time <= current_time <= end_time:
                self.class_status_label.text = "✅ CLASS IN SESSION"
                self.class_status_label.color = (0.2, 0.8, 0.2, 1)
            elif current_time < start_time:
                self.class_status_label.text = "⏳ UPCOMING CLASS"
                self.class_status_label.color = (1, 0.6, 0, 1)
            else:
                self.class_status_label.text = "⏹️ CLASS ENDED"
                self.class_status_label.color = (0.8, 0.2, 0.2, 1)
        else:
            self.current_class_info = None
            day_name = datetime.now().strftime('%A')
            self.course_label.text = f"📚 No Class Scheduled"
            self.time_label.text = f"🕐 {day_name} - Free Time"
            self.room_label.text = f"🏫 Room: N/A"
            self.instructor_label.text = f"👨‍🏫 No Instructor"
            self.current_time_label.text = f"⏰ Current Time: {datetime.now().strftime('%I:%M:%S %p')}"
            self.class_status_label.text = "📅 NO CLASS"
            self.class_status_label.color = (0.5, 0.5, 0.5, 1)
    
    def create_ui(self):
        """Create the enhanced user interface"""
        # Header
        header = BoxLayout(size_hint=(1, 0.08), padding=10)
        with header.canvas.before:
            Color(0.2, 0.4, 0.8, 1)
            self.header_rect = Rectangle(pos=header.pos, size=header.size)
        header.bind(pos=self.update_rect, size=self.update_rect)
        
        header_label = Label(
            text='🎓 BAI-4 Attendance System - Spring 2024',
            font_size='24sp',
            bold=True,
            color=(1, 1, 1, 1)
        )
        header.add_widget(header_label)
        self.add_widget(header)
        
        # Registration Number Input Panel
        reg_panel = BoxLayout(size_hint=(1, 0.10), padding=10, spacing=10)
        with reg_panel.canvas.before:
            Color(0.95, 0.98, 1, 1)
            self.reg_panel_rect = RoundedRectangle(pos=reg_panel.pos, size=reg_panel.size, radius=[10])
        reg_panel.bind(pos=self.update_reg_rect, size=self.update_reg_rect)
        
        reg_label = Label(
            text='🎓 Registration Number:',
            size_hint=(0.3, 1),
            font_size='16sp',
            bold=True,
            color=(0.2, 0.2, 0.2, 1),
            halign='right',
            valign='middle'
        )
        reg_label.bind(size=reg_label.setter('text_size'))
        
        self.reg_input = TextInput(
            text='',
            hint_text='Enter your Registration Number (e.g., BAI4-001)',
            size_hint=(0.5, 1),
            multiline=False,
            font_size='18sp',
            background_color=(1, 1, 1, 1),
            foreground_color=(0.2, 0.2, 0.2, 1),
            padding=[15, 15],
            cursor_color=(0.2, 0.4, 0.8, 1)
        )
        
        verify_btn = Button(
            text='✓ Verify',
            size_hint=(0.2, 1),
            background_normal='',
            background_color=(0.3, 0.7, 0.4, 1),
            font_size='16sp',
            bold=True,
            color=(1, 1, 1, 1)
        )
        verify_btn.bind(on_press=self.verify_registration)
        
        reg_panel.add_widget(reg_label)
        reg_panel.add_widget(self.reg_input)
        reg_panel.add_widget(verify_btn)
        self.add_widget(reg_panel)
        
        # Current Class Information Panel
        class_panel = BoxLayout(size_hint=(1, 0.16), padding=10, spacing=5, orientation='vertical')
        with class_panel.canvas.before:
            Color(1, 1, 1, 1)
            self.class_panel_rect = RoundedRectangle(pos=class_panel.pos, size=class_panel.size, radius=[10])
        class_panel.bind(pos=self.update_class_rect, size=self.update_class_rect)
        
        # First row: Course and Status
        row1 = BoxLayout(size_hint=(1, 0.4), spacing=10)
        self.course_label = Label(
            text='📚 Loading...',
            font_size='16sp',
            bold=True,
            color=(0.2, 0.2, 0.2, 1),
            halign='left',
            valign='middle'
        )
        self.course_label.bind(size=self.course_label.setter('text_size'))
        
        self.class_status_label = Label(
            text='⏳ LOADING',
            font_size='15sp',
            bold=True,
            color=(1, 0.6, 0, 1),
            halign='right',
            valign='middle',
            size_hint=(0.4, 1)
        )
        self.class_status_label.bind(size=self.class_status_label.setter('text_size'))
        
        row1.add_widget(self.course_label)
        row1.add_widget(self.class_status_label)
        
        # Second row: Time and Current Time
        row2 = BoxLayout(size_hint=(1, 0.3), spacing=10)
        self.time_label = Label(
            text='🕐 --:-- - --:--',
            font_size='14sp',
            color=(0.3, 0.3, 0.3, 1),
            halign='left',
            valign='middle'
        )
        self.time_label.bind(size=self.time_label.setter('text_size'))
        
        self.current_time_label = Label(
            text='⏰ Current Time: --:--:--',
            font_size='14sp',
            color=(0.3, 0.3, 0.3, 1),
            halign='right',
            valign='middle',
            size_hint=(0.5, 1)
        )
        self.current_time_label.bind(size=self.current_time_label.setter('text_size'))
        
        row2.add_widget(self.time_label)
        row2.add_widget(self.current_time_label)
        
        # Third row: Room and Instructor
        row3 = BoxLayout(size_hint=(1, 0.3), spacing=10)
        self.room_label = Label(
            text='🏫 Room: ---',
            font_size='14sp',
            color=(0.3, 0.3, 0.3, 1),
            halign='left',
            valign='middle'
        )
        self.room_label.bind(size=self.room_label.setter('text_size'))
        
        self.instructor_label = Label(
            text='👨‍🏫 Instructor: ---',
            font_size='14sp',
            color=(0.3, 0.3, 0.3, 1),
            halign='right',
            valign='middle'
        )
        self.instructor_label.bind(size=self.instructor_label.setter('text_size'))
        
        row3.add_widget(self.room_label)
        row3.add_widget(self.instructor_label)
        
        class_panel.add_widget(row1)
        class_panel.add_widget(row2)
        class_panel.add_widget(row3)
        self.add_widget(class_panel)
        
        # Camera feed with border
        camera_container = BoxLayout(size_hint=(1, 0.45), padding=5)
        with camera_container.canvas.before:
            Color(1, 1, 1, 1)
            self.camera_container_rect = RoundedRectangle(pos=camera_container.pos, size=camera_container.size, radius=[10])
        camera_container.bind(pos=self.update_camera_rect, size=self.update_camera_rect)
        
        self.img = Image()
        camera_container.add_widget(self.img)
        self.add_widget(camera_container)
        
        # Status label
        self.status_label = Label(
            text='📹 Position your face in front of camera and click "Capture Attendance"',
            size_hint=(1, 0.08),
            font_size='16sp',
            color=(0.3, 0.3, 0.3, 1),
            halign='center',
            valign='middle',
            bold=True
        )
        self.status_label.bind(size=self.status_label.setter('text_size'))
        self.add_widget(self.status_label)
        
        # Attendance info panel
        attendance_panel = BoxLayout(size_hint=(1, 0.08), padding=10)
        with attendance_panel.canvas.before:
            Color(0.9, 0.95, 1, 1)
            self.attendance_panel_rect = RoundedRectangle(pos=attendance_panel.pos, size=attendance_panel.size, radius=[8])
        attendance_panel.bind(pos=self.update_attendance_rect, size=self.update_attendance_rect)
        
        self.attendance_label = Label(
            text='✓ Today\'s Attendance: None',
            font_size='15sp',
            color=(0.2, 0.4, 0.8, 1),
            bold=True
        )
        attendance_panel.add_widget(self.attendance_label)
        self.add_widget(attendance_panel)
        
        # Button layout
        btn_layout = BoxLayout(size_hint=(1, 0.11), spacing=10, padding=10)
        
        # Capture button
        self.capture_btn = Button(
            text='📸 Capture Attendance',
            background_normal='',
            background_color=(0.2, 0.6, 1, 1),
            font_size='17sp',
            bold=True,
            color=(1, 1, 1, 1)
        )
        self.capture_btn.bind(on_press=self.capture_and_recognize)
        btn_layout.add_widget(self.capture_btn)
        
        # Reset button
        reset_btn = Button(
            text='🔄 Reset',
            background_normal='',
            background_color=(1, 0.4, 0.4, 1),
            font_size='16sp',
            bold=True,
            color=(1, 1, 1, 1)
        )
        reset_btn.bind(on_press=self.reset_attendance)
        btn_layout.add_widget(reset_btn)
        
        # View attendance button
        view_btn = Button(
            text='📋 View Records',
            background_normal='',
            background_color=(0.3, 0.7, 0.4, 1),
            font_size='16sp',
            bold=True,
            color=(1, 1, 1, 1)
        )
        view_btn.bind(on_press=self.view_attendance)
        btn_layout.add_widget(view_btn)
        
        self.add_widget(btn_layout)
    
    def update_reg_rect(self, instance, value):
        self.reg_panel_rect.pos = instance.pos
        self.reg_panel_rect.size = instance.size
    
    def update_rect(self, instance, value):
        self.header_rect.pos = instance.pos
        self.header_rect.size = instance.size
    
    def update_class_rect(self, instance, value):
        self.class_panel_rect.pos = instance.pos
        self.class_panel_rect.size = instance.size
    
    def update_camera_rect(self, instance, value):
        self.camera_container_rect.pos = instance.pos
        self.camera_container_rect.size = instance.size
    
    def update_attendance_rect(self, instance, value):
        self.attendance_panel_rect.pos = instance.pos
        self.attendance_panel_rect.size = instance.size
    
    def verify_registration(self, instance):
        """Verify the registration number"""
        reg_no = self.reg_input.text.strip()
        
        if not reg_no:
            self.status_label.text = '⚠️ Please enter your registration number!'
            return
        
        # Store the registration number temporarily for the next capture
        self.status_label.text = f'✓ Registration number {reg_no} verified! Ready to capture attendance.'
        print(f"Registration number verified: {reg_no}")
    
    def predict_face(self, face_rgb):
        """Predict the identity of a face"""
        try:
            face_resized = cv2.resize(face_rgb, (160, 160))
            face_expanded = np.expand_dims(face_resized, axis=0)
            embedding = self.facenet.embeddings(face_expanded)
            prediction = self.svm_model.predict(embedding)
            probability = self.svm_model.predict_proba(embedding)
            predicted_label = self.encoder.inverse_transform(prediction)[0]
            confidence = np.max(probability) * 100
            return predicted_label, confidence
        except Exception as e:
            print(f"Error in prediction: {e}")
            return None, 0.0
    
    def check_attendance_status(self, class_start_time):
        """Check if student is on time, late, or very late"""
        current_time = datetime.now().time()
        try:
            start_time_obj = datetime.strptime(class_start_time, '%H:%M').time()
            
            time_diff = datetime.combine(datetime.today(), current_time) - datetime.combine(datetime.today(), start_time_obj)
            minutes_late = time_diff.total_seconds() / 60
            
            if minutes_late <= 0:
                return "On Time"
            elif minutes_late <= 15:
                return "Late"
            else:
                return "Very Late"
        except:
            return "On Time"
    
    def mark_attendance(self, name, confidence):
        if not self.current_class_info:
            return None, None
        
        # Get registration number
        reg_no = self.reg_input.text.strip()
        if not reg_no:
            self.status_label.text = '⚠️ Please enter your registration number first!'
            return None, None
        
        current_time = datetime.now()
        date_str = current_time.strftime('%Y-%m-%d')
        time_str = current_time.strftime('%I:%M:%S %p')
        
        # Get full name from database or use detected name
        full_name = self.student_database.get(name.lower(), {}).get('name', name)
        
        # Update student database with registration number
        if name.lower() in self.student_database:
            self.student_database[name.lower()]['reg_no'] = reg_no
        
        class_info = self.current_class_info
        status = self.check_attendance_status(class_info['start'])
        
        with open(self.attendance_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                reg_no,
                full_name, 
                class_info['course'], 
                class_info['room'],
                class_info['instructor'],
                date_str, 
                time_str, 
                class_info['start'], 
                class_info['end'], 
                f"{confidence:.2f}%", 
                status
            ])
        
        self.attendance_marked.add(f"{full_name} ({reg_no})")
        print(f"Attendance marked for {full_name} (Reg: {reg_no}) in {class_info['course']} - Status: {status}")
        return full_name, status
    
    def capture_and_recognize(self, instance):
        if self.processing:
            return
        
        # Check registration number first
        if not self.reg_input.text.strip():
            self.status_label.text = '⚠️ Please enter your registration number first!'
            return
        
        if not self.current_class_info:
            self.status_label.text = '⚠️ No class scheduled at this time! Cannot mark attendance.'
            return
        
        if self.current_frame is None:
            self.status_label.text = '⚠️ No camera feed available'
            return
        
        self.processing = True
        self.capture_btn.disabled = True
        self.status_label.text = '⏳ Processing... Please wait'
        Clock.schedule_once(self.process_capture, 0.1)
    
    def process_capture(self, dt):
        try:
            frame_rgb = self.current_frame.copy()
            detections = self.detector.detect_faces(frame_rgb)
            
            if not detections:
                self.status_label.text = '❌ No face detected! Please position your face clearly and try again.'
                self.processing = False
                self.capture_btn.disabled = False
                return
            
            detection = max(detections, key=lambda x: x['confidence'])
            x1, y1, w, h = detection['box']
            x2, y2 = x1 + w, y1 + h
            x1, y1 = max(0, x1), max(0, y1)
            x2 = min(frame_rgb.shape[1], x2)
            y2 = min(frame_rgb.shape[0], y2)
            
            face = frame_rgb[y1:y2, x1:x2]
            
            if face.size == 0:
                self.status_label.text = '❌ Face extraction failed! Please try again.'
                self.processing = False
                self.capture_btn.disabled = False
                return
            
            name, confidence = self.predict_face(face)
            
            if name and confidence > 80:
                full_name, status = self.mark_attendance(name, confidence)
                
                if full_name and status:
                    status_emoji = "✅" if status == "On Time" else "⏰" if status == "Late" else "⚠️"
                    self.status_label.text = f'{status_emoji} SUCCESS! Attendance marked for {full_name} ({status}) - Confidence: {confidence:.1f}%'
                    
                    color = (0, 255, 0) if status == "On Time" else (255, 165, 0) if status == "Late" else (255, 0, 0)
                    cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), color, 3)
                    text = f'{full_name} - {confidence:.1f}%'
                    cv2.putText(frame_rgb, text, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    cv2.putText(frame_rgb, status, (x1, y2 + 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    self.display_frame(frame_rgb)
                else:
                    self.status_label.text = '⚠️ Error marking attendance'
            else:
                self.status_label.text = f'❌ ATTENDANCE NOT MARKED! Low confidence: {name} ({confidence:.1f}%). Required: >80%'
                color = (255, 0, 0)
                cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), color, 2)
                text = f'{name} - {confidence:.1f}% (Too Low)'
                cv2.putText(frame_rgb, text, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                self.display_frame(frame_rgb)
            
            if self.attendance_marked:
                course_short = self.current_class_info['course'].split('-')[0].strip() if self.current_class_info else 'N/A'
                self.attendance_label.text = f'✓ {course_short} Attendance ({len(self.attendance_marked)}): {", ".join(self.attendance_marked)}'
        
        except Exception as e:
            print(f"Error during processing: {e}")
            self.status_label.text = f'❌ Error: {str(e)}'
        finally:
            self.processing = False
            self.capture_btn.disabled = False
    
    def update_camera(self, dt):
        if self.processing:
            return
        ret, frame = self.capture.read()
        if not ret:
            return
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.current_frame = frame_rgb
        self.display_frame(frame_rgb)
    
    def display_frame(self, frame_rgb):
        buf = cv2.flip(frame_rgb, 0).tobytes()
        texture = Texture.create(size=(frame_rgb.shape[1], frame_rgb.shape[0]), colorfmt='rgb')
        texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
        self.img.texture = texture
    
    def reset_attendance(self, instance):
        self.attendance_marked.clear()
        self.reg_input.text = ''
        self.status_label.text = '🔄 Attendance reset! Ready to capture new attendance.'
        self.attendance_label.text = '✓ Today\'s Attendance: None'
        print("Attendance reset!")
    
    def view_attendance(self, instance):
        print("\n" + "="*100)
        print("📋 BAI-4 ATTENDANCE RECORD - SPRING 2024")
        print("="*100)
        try:
            with open(self.attendance_file, 'r') as f:
                reader = csv.reader(f)
                for i, row in enumerate(reader):
                    if i == 0:
                        print(" | ".join([f"{col:^15}" for col in row]))
                        print("-"*100)
                    else:
                        print(" | ".join([f"{col:^15}" for col in row]))
        except Exception as e:
            print(f"Error reading attendance: {e}")
        print("="*100 + "\n")
    
    def on_stop(self):
        self.capture.release()


class FaceAttendanceApp(App):
    def build(self):
        self.title = 'BAI-4 Face Recognition Attendance System - Spring 2024'
        return AttendanceSystem()
    
    def on_stop(self):
        self.root.on_stop()


if __name__ == '__main__':
    FaceAttendanceApp().run()