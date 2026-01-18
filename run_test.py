import os
import math
import traceback
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from cvzone.HandTrackingModule import HandDetector
import tkinter as tk
from PIL import Image, ImageTk

# configuration
OFFSET = 29
MODEL_PATH = 'cnn_model_mediapipe.keras'

print(f"DEBUG: Script started. Looking for model at: {os.path.abspath(MODEL_PATH)}")

# detectors
try:
    hd = HandDetector(maxHands=1)
    hd2 = HandDetector(maxHands=1)
    print("DEBUG: HandDetectors initialized successfully.")
except Exception as e:
    print(f"DEBUG: Error initializing HandDetectors: {e}")
    traceback.print_exc()

class Application:
    def __init__(self):
        try:
            print("DEBUG: Initializing Application...")
            self.vs = cv2.VideoCapture(0)
            if not self.vs.isOpened():
                print("DEBUG: ERROR - Could not open video source (webcam).")
            else:
                print("DEBUG: Webcam opened successfully.")

            print(f"DEBUG: Attempting to load model from {MODEL_PATH}...")
            # Using compile=False often helps with loading issues in different environments
            self.model = load_model(MODEL_PATH, compile=False)
            print("DEBUG: Model loaded successfully.")

            self.current_symbol = ' '
            self.pts = None

            # GUI
            print("DEBUG: Setting up GUI...")
            self.root = tk.Tk()
            self.root.title('ASL — Current Character')
            self.root.protocol('WM_DELETE_WINDOW', self.destructor)
            self.root.geometry('1300x700')

            # ================== MAIN CONTAINER ==================
            main_frame = tk.Frame(self.root, bg="white")
            main_frame.pack(fill="both", expand=True, padx=10, pady=10)

            # configure grid
            main_frame.columnconfigure(0, weight=4)
            main_frame.columnconfigure(1, weight=3)
            main_frame.columnconfigure(2, weight=2)
            main_frame.rowconfigure(0, weight=1)

            # ================== LEFT: WEBCAM ==================
            left_frame = tk.Frame(main_frame, bd=2, relief="groove")
            left_frame.grid(row=0, column=0, padx=5, pady=5, sticky="n")

            self.panel = tk.Label(left_frame)
            self.panel.pack()

            # ================== CENTER: PROCESSED + LETTER ==================
            center_frame = tk.Frame(main_frame, bd=2, relief="groove")
            center_frame.grid(row=0, column=1, padx=5, pady=5, sticky="n")

            self.panel2 = tk.Label(center_frame)
            self.panel2.pack(pady=10)

            self.char_label = tk.Label(
                center_frame,
                text=" ",
                font=("Helvetica", 72, "bold")
            )
            self.char_label.pack(pady=5)

            # ================== RIGHT: STATIC ASL IMAGE ==================
            right_frame = tk.Frame(main_frame, bd=2, relief="groove")
            right_frame.grid(row=0, column=2, padx=5, pady=5, sticky="n")

            self.panel3 = tk.Label(right_frame)
            self.panel3.pack()

            print("DEBUG: Loading signs.png...")
            if os.path.exists('signs.png'):
                img_static = Image.open('signs.png')
                img_static = img_static.resize((300, 400), Image.LANCZOS)
                self.imgtk3 = ImageTk.PhotoImage(img_static)
                self.panel3.config(image=self.imgtk3)
                print("DEBUG: signs.png loaded and displayed.")
            else:
                print("DEBUG: Warning - signs.png not found.")

            print("DEBUG: Starting video_loop...")
            self.video_loop()

        except Exception as e:
            print(f"DEBUG: Exception in Application.__init__: {e}")
            traceback.print_exc()

    def video_loop(self):
        try:
            ok, frame = self.vs.read()
            if not ok:
                # Use a small print for frame failure but don't flood the console
                self.root.after(10, self.video_loop)
                return

            frame = cv2.flip(frame, 1)
            
            # Use try-except specifically for hand detection
            try:
                hands = hd.findHands(frame, draw=False, flipType=True)
            except Exception as e:
                print(f"DEBUG: Error in first HandDetector (hd): {e}")
                hands = None

            # show webcam frame
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image).resize((640, 480))
            imgtk = ImageTk.PhotoImage(image=img)
            self.panel.imgtk = imgtk
            self.panel.config(image=imgtk)

            if hands:
                hand = hands[0]
                x, y, w, h = hand['bbox']

                # crop safely
                h_img, w_img = frame.shape[:2]
                x1 = max(0, x - OFFSET)
                y1 = max(0, y - OFFSET)
                x2 = min(w_img, x + w + OFFSET)
                y2 = min(h_img, y + h + OFFSET)
                crop = frame[y1:y2, x1:x2]

                # processed white 400x400 canvas
                white = 255 * np.ones((400, 400, 3), dtype=np.uint8)

                # detect landmarks again on the crop
                try:
                    handz = hd2.findHands(crop, draw=False, flipType=True)
                except Exception as e:
                    print(f"DEBUG: Error in second HandDetector (hd2): {e}")
                    handz = None

                if handz:
                    h2 = handz[0]
                    self.pts = h2['lmList']

                    # center offsets to draw into 400x400 canvas
                    os_x = ((400 - (x2 - x1)) // 2) - 15
                    os_y = ((400 - (y2 - y1)) // 2) - 15

                    # draw skeleton and points (kept from original logic)
                    # Skeleton lines
                    for t in range(0, 4):
                        cv2.line(white, (self.pts[t][0] + os_x, self.pts[t][1] + os_y), (self.pts[t + 1][0] + os_x, self.pts[t + 1][1] + os_y), (0, 255, 0), 3)
                    for t in range(5, 8):
                        cv2.line(white, (self.pts[t][0] + os_x, self.pts[t][1] + os_y), (self.pts[t + 1][0] + os_x, self.pts[t + 1][1] + os_y), (0, 255, 0), 3)
                    for t in range(9, 12):
                        cv2.line(white, (self.pts[t][0] + os_x, self.pts[t][1] + os_y), (self.pts[t + 1][0] + os_x, self.pts[t + 1][1] + os_y), (0, 255, 0), 3)
                    for t in range(13, 16):
                        cv2.line(white, (self.pts[t][0] + os_x, self.pts[t][1] + os_y), (self.pts[t + 1][0] + os_x, self.pts[t + 1][1] + os_y), (0, 255, 0), 3)
                    for t in range(17, 20):
                        cv2.line(white, (self.pts[t][0] + os_x, self.pts[t][1] + os_y), (self.pts[t + 1][0] + os_x, self.pts[t + 1][1] + os_y), (0, 255, 0), 3)

                    # Connection lines
                    cv2.line(white, (self.pts[5][0] + os_x, self.pts[5][1] + os_y), (self.pts[9][0] + os_x, self.pts[9][1] + os_y), (0, 255, 0), 3)
                    cv2.line(white, (self.pts[9][0] + os_x, self.pts[9][1] + os_y), (self.pts[13][0] + os_x, self.pts[13][1] + os_y), (0, 255, 0), 3)
                    cv2.line(white, (self.pts[13][0] + os_x, self.pts[13][1] + os_y), (self.pts[17][0] + os_x, self.pts[17][1] + os_y), (0, 255, 0), 3)
                    cv2.line(white, (self.pts[0][0] + os_x, self.pts[0][1] + os_y), (self.pts[5][0] + os_x, self.pts[5][1] + os_y), (0, 255, 0), 3)
                    cv2.line(white, (self.pts[0][0] + os_x, self.pts[0][1] + os_y), (self.pts[17][0] + os_x, self.pts[17][1] + os_y), (0, 255, 0), 3)

                    # Keypoints
                    for i in range(21):
                        cv2.circle(white, (self.pts[i][0] + os_x, self.pts[i][1] + os_y), 2, (0, 0, 255), 1)

                    # predict and update
                    try:
                        self.predict(white)
                    except Exception as e:
                        print(f"DEBUG: Prediction error: {e}")

                    # show processed image
                    proc_img = Image.fromarray(cv2.cvtColor(white, cv2.COLOR_BGR2RGB))
                    imgtk2 = ImageTk.PhotoImage(image=proc_img)
                    self.panel2.imgtk = imgtk2
                    self.panel2.config(image=imgtk2)

                    # update current character label
                    self.char_label.config(text=str(self.current_symbol))

        except Exception as e:
            print("DEBUG: General exception in video_loop:", e)
            print('DEBUG Traceback:', traceback.format_exc())
        finally:
            self.root.after(10, self.video_loop)

    def distance(self, x, y):
        return math.sqrt(((x[0] - y[0]) ** 2) + ((x[1] - y[1]) ** 2))

    def predict(self, img_400):
        try:
            img = cv2.resize(img_400, (224, 224))
            img = img / 255.0
            img = img.reshape(1, 224, 224, 3)

            pred = self.model.predict(img, verbose=0)
            idx = int(np.argmax(pred))
            
            # Check if index is within expected range (A-Z is 0-25)
            self.current_symbol = chr(ord('A') + idx)
        except Exception as e:
            print(f"DEBUG: predict() function error: {e}")
            self.current_symbol = "Error"

    def destructor(self):
        print("DEBUG: Shutting down application...")
        try:
            self.root.destroy()
        except Exception:
            pass
        try:
            self.vs.release()
        except Exception:
            pass
        cv2.destroyAllWindows()
        print("DEBUG: Cleanup complete.")

if __name__ == '__main__':
    print('Starting minimal ASL viewer...')
    try:
        app = Application()
        app.root.mainloop()
    except Exception as e:
        print(f"DEBUG: Fatal error in __main__: {e}")
        traceback.print_exc()