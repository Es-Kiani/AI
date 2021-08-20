import cv2


class VideoDetector:
    def __init__(self, got_xml):
        self.trained_Data_xml = got_xml
        try:
            self.trained_Data = cv2.CascadeClassifier(
                cv2.data.haarcascades + self.trained_Data_xml)
        except:
            raise ValueError("Input xml File Could Not Open")

    def startDetection(self, videoFile, type='r'):
        self.video = cv2.VideoCapture(videoFile)

        while 1:
            self.isSuccessfullRead, self.frame_RGB = self.video.read()
            if (self.isSuccessfullRead == False):
                raise ValueError("Could Not Read Video...!")

            self.frame_GRY = cv2.cvtColor(self.frame_RGB, cv2.COLOR_BGR2GRAY)
            self.face_Cords = self.trained_Data.detectMultiScale(
                self.frame_GRY)

            if type == 'r':
                for (x, y, w, h) in self.face_Cords:
                    cv2.rectangle(self.frame_RGB,
                                  (x, y), (x+w, y+h), (0, 255, 0), 2)
            elif type == "c":
                for (x, y, w, h) in self.face_Cords:
                    cv2.circle(self.frame_RGB,
                               (x+w//2, y+h//2), ((w//2) + 5), (255, 50, 200), 2)
            else:
                raise TypeError("Your Detection Frame Type Is Not Allowed!")

            cv2.imshow("Video Face Detector", self.frame_RGB)

            self.key = cv2.waitKey(1)
            if self.key == 81 or self.key == 113:
                break
        self.video.release()
        cv2.destroyAllWindows()
        print("\nRun Successfull.")
        print("\nAll Windows's Are Destroyed.")