import cv2 as cv


class LiveDetector:
    def __init__(self, faceXML, smileXML):
        try:
            self.face_Detector = cv.CascadeClassifier(
                cv.data.haarcascades + faceXML)
            self.smile_Detector = cv.CascadeClassifier(
                cv.data.haarcascades + smileXML)
        except:
            raise FileNotFoundError(
                "Could Not Open XML Files...!")

    def start(self, webcamURL=0, faceScaleFactor=1.1, faceMinNeighbors=1, smileScaleFactor=1.1, smileMinNeighbors=1):
        self.webcam = self._connect_Webcam(webcamURL)

        while True:
            try:
                self.isRead, self.RGB_Frame = self.webcam.read()
                if not self.isRead:
                    raise ConnectionAbortedError(
                        "Connection To Your Webcam Is Abort...!")
                self.GRY_Frame = cv.cvtColor(self.RGB_Frame, cv.COLOR_BGR2GRAY)

                for x, y, w, h in self._Get_FaceCoords(self.GRY_Frame, faceScaleFactor, faceMinNeighbors):
                    cv.rectangle(self.RGB_Frame,
                                 (x, y), (x+w, y+h), (0, 255, 0), 2)

                    if self._isExistSmile(self.GRY_Frame[x:x+w, y:y+h], smileScaleFactor, smileMinNeighbors):
                        cv.putText(self.RGB_Frame, "Smiling",
                                   (x, y+h+25), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0))

                cv.imshow("Live Webcam Smile Detector", self.RGB_Frame)

                self.key = cv.waitKey(1)
                if self.key == 81 or self.key == 113:
                    break
            except:
                raise RuntimeError("An Error Occurred When Detecting...!")

        self.webcam.release()
        cv.destroyAllWindows()

    def _connect_Webcam(self, webcamURL):
        try:
            self.webcam = cv.VideoCapture(webcamURL)
            return self.webcam
        except:
            raise ConnectionAbortedError(
                "Connection To Your Webcam Is Abort...!")

    def _Get_FaceCoords(self, frame, scaleFactor, minNeighbors):
        self.face_Coords = self.face_Detector.detectMultiScale(
            frame, scaleFactor=scaleFactor, minNeighbors=minNeighbors)
        return self.face_Coords

    def _isExistSmile(self, face, scaleFactor, minNeighbors):
        self.smile_Coords = self.smile_Detector.detectMultiScale(
            face, scaleFactor=scaleFactor, minNeighbors=minNeighbors)
        return False if len(self.smile_Coords) == 0 else True
