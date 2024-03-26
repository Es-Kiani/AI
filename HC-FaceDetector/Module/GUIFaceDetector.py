from typing import Any
from Module.Singleton import Singleton
import PIL.Image
from PIL import ImageTk
import tkinter
from tkinter import *
from tkinter import Canvas
from tkinter import Message, messagebox, ttk
import cv2 as cv
from threading import Thread


class GUIFaceDetector(Singleton):
    def __init__(self, webcamURL=0, faceXML="haarcascade_frontalface_default.xml") -> None:
        super().__init__()
        self._detecting = False
        self.webcamURL = webcamURL
        self.faceXML = faceXML

    def _GUI_Builder(self) -> None:
        self._window = Tk()

        self._window.title("Live Webcam Face Detector")
        self._window.resizable(False, False)

        if self.webcamURL == 0:
            self._window.geometry("650x510")
            self._label = tkinter.Canvas(self._window, width=640, height=420)
        else:
            self._window.geometry("480x685")
            self._label = tkinter.Canvas(self._window, width=480, height=570)

        self._copyright = Label(
            self._window, text="©2021 Esfandiar Kiani, All rights reserved.")
        self._startBtn = ttk.Button(self._window)
        self._stopBtn = ttk.Button(self._window)

        self._label.create_image(0, 0, anchor=tkinter.CENTER)
        self._startBtn.configure(
            text="Start Detecting", cursor="hand2", command=self._start_Detecting)
        self._stopBtn.configure(
            text="Stop Detecting", cursor="hand2", command=self._stop_Detecting)

        self._label.pack(side="top", anchor=tkinter.N)
        self._startBtn.pack(expand="yes")
        self._stopBtn.pack(expand="yes")
        self._copyright.pack(side="bottom", anchor=tkinter.S)

        self._window.mainloop()
        return

    def _faceDetector_Builder(self) -> None:
        if self._detecting:
            try:
                detector = cv.CascadeClassifier(
                    cv.data.haarcascades + self.faceXML)

                webcam = cv.VideoCapture(self.webcamURL)
                is_Webcam_Connect = True

                while self._detecting:
                    is_Read, BGR_Frame = webcam.read()
                    if not is_Read:
                        messagebox.showerror(
                            "Connection Error", "Could Not Connect To Your Webcam!")
                        raise ConnectionError(
                            "[-Connection Error] Could Not Connect To Your Webcam!")

                    RGB_Frame = cv.cvtColor(BGR_Frame, cv.COLOR_BGR2RGB)
                    GRY_Frame = cv.cvtColor(BGR_Frame, cv.COLOR_BGR2GRAY)

                    face_Coords = detector.detectMultiScale(
                        GRY_Frame, scaleFactor=1.4, minNeighbors=3)
                    for x, y, w, h in face_Coords:
                        cv.rectangle(RGB_Frame,
                                     (x, y), (x+w, y+h), (0, 255, 0), 2)

                    image = PIL.Image.fromarray(RGB_Frame)
                    new_Frame = ImageTk.PhotoImage(image)
                    self._label.create_image(
                        0, 0, image=new_Frame, anchor=tkinter.NW)

                    self._label.update()
            except ConnectionError:
                pass
            except:
                messagebox.showwarning(
                    "Connection Aborted Error", "Webcam Connection Was Aborted...!")
                raise ConnectionAbortedError(
                    "[-Connection Aborted Error] Webcam Connection Was Aborted...!")
            finally:
                self._detecting = False
                if is_Webcam_Connect:
                    webcam.release()
        return

    def _start_Thread(self, function, daemon=False) -> Thread:
        thread = Thread(target=function, daemon=daemon)
        if thread.is_alive():
            return
        thread.setName(f"{function.__name__}")
        thread.start()
        return thread

    def _start_Detecting(self) -> None:
        self._detecting = True
        self._start_Thread(self._faceDetector_Builder, True)
        return

    def _stop_Detecting(self) -> None:
        if self._detecting:
            self._detecting = False
        else:
            self._window.destroy()
        return

    def run(self) -> None:
        self._start_Thread(self._GUI_Builder)
        return

    def __del__(self):
        self._stop_Detecting()
        return
