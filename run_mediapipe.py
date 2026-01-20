import os
import math
import traceback
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from cvzone.HandTrackingModule import HandDetector
import tkinter as tk
from PIL import Image, ImageTk
import time
import json


# configuration
OFFSET = 29
MODEL_PATH = 'net_model_mediapipe.keras'

# detectors
hd = HandDetector(maxHands=1)
hd2 = HandDetector(maxHands=1)

class Application:
    def __init__(self):
        self.vs = cv2.VideoCapture(0)
        self.model = load_model(MODEL_PATH)
        self.current_symbol = ' '
        self.pts = None
        self.prev_time = time.time()
        self.fps = 0
        self.frame_count = 0
        self.fps_log = []
        self.start_benchmark = time.time()
        self.benchmark_duration = 30  # seconds
        self.benchmark_saved = False
        self.hand_detected = False
        with open("classes_labels.json", "r") as f:
            self.class_labels = json.load(f)

        # GUI
        self.root = tk.Tk()
        self.root.title('ASL — Current Character')
        self.root.protocol('WM_DELETE_WINDOW', self.destructor)
        self.root.geometry('1300x700')

        # # camera preview
        # self.panel = tk.Label(self.root)
        # self.panel.place(x=40, y=10, width=640, height=480)

        # # processed hand image / prediction (center)
        # self.panel2 = tk.Label(self.root)
        # # a bit larger so the skeleton is visible
        # self.panel2.place(x=720, y=10, width=300, height=300)

        # # current character big label (below the prediction panel)
        # self.char_label = tk.Label(self.root, text=' ', font=('Helvetica', 72, 'bold'))
        # self.char_label.place(x=720, y=420, width=300, height=120)

        # # NEW: panel to show static image (signs.png) on the right
        # self.panel3 = tk.Label(self.root)
        # self.panel3.place(x=1150, y=110, width=500, height=400)  # fits to the right

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



        # img_static = Image.open('signs.png')
        # img_static = img_static.resize((500, 400), Image.ANTIALIAS)
        # self.imgtk3 = ImageTk.PhotoImage(image=img_static)
        # self.panel3.imgtk = self.imgtk3
        # self.panel3.config(image=self.imgtk3)

        img_static = Image.open('signs.png')
        img_static = img_static.resize((300, 400), Image.LANCZOS)
        self.imgtk3 = ImageTk.PhotoImage(img_static)
        self.panel3.config(image=self.imgtk3)


        
        self.video_loop()

    def video_loop(self):
        try:
            self.hand_detected = False
            ok, frame = self.vs.read()
            if not ok:
                self.root.after(10, self.video_loop)
                return

            frame = cv2.flip(frame, 1)
            cv2.putText(
                frame,
                f"FPS: {self.fps:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            hands = hd.findHands(frame, draw=False, flipType=True)

            # show webcam frame
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #img = Image.fromarray(cv2image)
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
                handz = hd2.findHands(crop, draw=False, flipType=True)
                if handz:
                    self.hand_detected = True
                    h2 = handz[0]
                    self.pts = h2['lmList']

                    # center offsets to draw into 400x400 canvas
                    os_x = ((400 - (x2 - x1)) // 2) - 15
                    os_y = ((400 - (y2 - y1)) // 2) - 15

                    # draw skeleton and points (kept from original logic)
                    for t in range(0, 4):
                        cv2.line(white,
                                 (self.pts[t][0] + os_x, self.pts[t][1] + os_y),
                                 (self.pts[t + 1][0] + os_x, self.pts[t + 1][1] + os_y),  (0, 255, 0), 3)
                    for t in range(5, 8):
                        cv2.line(white, (self.pts[t][0] + os_x, self.pts[t][1] + os_y), (self.pts[t + 1][0] + os_x, self.pts[t + 1][1] + os_y), (0, 255, 0), 3)
                    for t in range(9, 12):
                        cv2.line(white, (self.pts[t][0] + os_x, self.pts[t][1] + os_y), (self.pts[t + 1][0] + os_x, self.pts[t + 1][1] + os_y), (0, 255, 0), 3)
                    for t in range(13, 16):
                        cv2.line(white, (self.pts[t][0] + os_x, self.pts[t][1] + os_y), (self.pts[t + 1][0] + os_x, self.pts[t + 1][1] + os_y), (0, 255, 0), 3)
                    for t in range(17, 20):
                        cv2.line(white, (self.pts[t][0] + os_x, self.pts[t][1] + os_y), (self.pts[t + 1][0] + os_x, self.pts[t + 1][1] + os_y), (0, 255, 0), 3)

                    cv2.line(white, (self.pts[5][0] + os_x, self.pts[5][1] + os_y), (self.pts[9][0] + os_x, self.pts[9][1] + os_y), (0, 255, 0), 3)
                    cv2.line(white, (self.pts[9][0] + os_x, self.pts[9][1] + os_y), (self.pts[13][0] + os_x, self.pts[13][1] + os_y), (0, 255, 0), 3)
                    cv2.line(white, (self.pts[13][0] + os_x, self.pts[13][1] + os_y), (self.pts[17][0] + os_x, self.pts[17][1] + os_y), (0, 255, 0), 3)
                    cv2.line(white, (self.pts[0][0] + os_x, self.pts[0][1] + os_y), (self.pts[5][0] + os_x, self.pts[5][1] + os_y), (0, 255, 0), 3)
                    cv2.line(white, (self.pts[0][0] + os_x, self.pts[0][1] + os_y), (self.pts[17][0] + os_x, self.pts[17][1] + os_y), (0, 255, 0), 3)

                    for i in range(21):
                        cv2.circle(white, (self.pts[i][0] + os_x, self.pts[i][1] + os_y), 2, (0, 0, 255), 1)

                    # predict and update
                    self.predict(white)

                    # show processed image
                    proc_img = Image.fromarray(cv2.cvtColor(white, cv2.COLOR_BGR2RGB))
                    imgtk2 = ImageTk.PhotoImage(image=proc_img)
                    self.panel2.imgtk = imgtk2
                    self.panel2.config(image=imgtk2)

                    # update current character label
                    self.char_label.config(text=str(self.current_symbol))
            else:
                # AFTER all hand detection logic
                if not self.hand_detected:
                    cv2.putText(
                        frame,
                        "No hand detected",
                        (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 0, 255),
                        2
                    ) 
            # show webcam frame (convert to RGB and display in Tkinter)
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image).resize((640, 480))
            imgtk = ImageTk.PhotoImage(image=img)
            self.panel.imgtk = imgtk
            self.panel.config(image=imgtk)

            # ================= FPS calculation =================
            if self.hand_detected:
                self.frame_count += 1

            current_time = time.time()
            elapsed = current_time - self.prev_time

            if elapsed >= 1.0:
                self.fps = self.frame_count / elapsed
                print(f"FPS: {self.fps:.2f}")

                if self.hand_detected:
                    self.fps_log.append(self.fps)


                self.prev_time = current_time
                self.frame_count = 0

            # ================= Stop benchmark after fixed time =================
            if (not self.benchmark_saved and
                    time.time() - self.start_benchmark >= self.benchmark_duration):

                avg_fps = sum(self.fps_log) / len(self.fps_log) if self.fps_log else 0

                with open("fps_results.txt", "a") as f:
                    f.write("\n-----------------------------\n")
                    f.write(f"Average FPS over {self.benchmark_duration} seconds: {avg_fps:.2f}\n")
                    f.write("FPS values:\n")
                    for fps in self.fps_log:
                        f.write(f"{fps:.2f}\n")

                print("Benchmark completed.")
                print(f"Average FPS: {avg_fps:.2f}")

                self.benchmark_saved = True

            
        except Exception as e:
            print("Exception in video_loop:", e)
            print('==', traceback.format_exc())
        finally:
            self.root.after(10, self.video_loop)

    def distance(self, x, y):
        return math.sqrt(((x[0] - y[0]) ** 2) + ((x[1] - y[1]) ** 2))


    # def predict(self, img_400):
    #     img = cv2.resize(img_400, (224, 224))
    #     img = img / 255.0
    #     img = img.reshape(1, 224, 224, 3)

    #     pred = self.model.predict(img, verbose=0)
    #     idx = int(np.argmax(pred))

    #     self.current_symbol = chr(ord('A') + idx)

    def predict(self, img_400):
        img = cv2.resize(img_400, (224, 224))
        img = img / 255.0
        img = img.reshape(1, 224, 224, 3)

        pred = self.model.predict(img, verbose=0)
        idx = int(np.argmax(pred))

        # ✅ correct mapping
        self.current_symbol = self.class_labels[idx]



    def destructor(self):
        try:
            self.root.destroy()
        except Exception:
            pass
        try:
            self.vs.release()
        except Exception:
            pass
        cv2.destroyAllWindows()


if __name__ == '__main__':
    print('Starting minimal ASL viewer...')
    Application().root.mainloop()