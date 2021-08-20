import cv2


class VideoDetector:
    def __init__(self, got_xml):
        self.trained_Data_xml = got_xml
        try:
            self.car_Detector = cv2.CascadeClassifier(self.trained_Data_xml)
        except:
            raise ValueError("Input xml File Could Not Open")

    def startDetection(self, videoFile, type='r', color=(0, 255, 0), BD=2):
        self.video = cv2.VideoCapture(videoFile)

        while 1:
            self.isReadSuccessfull, self.frame_RGB = self.video.read()
            if (self.isReadSuccessfull == False):
                # raise ValueError("Could Not Read Video...!")
                return

            self.frame_GRY = cv2.cvtColor(self.frame_RGB, cv2.COLOR_BGR2GRAY)
            self.cars_Cords = self.car_Detector.detectMultiScale(
                self.frame_GRY, 1.1, 1)

            if type == 'r':
                for (x, y, w, h) in self.cars_Cords:
                    cv2.rectangle(self.frame_RGB, (x, y),
                                  (x+w, y+h), color, BD)
            elif type == "c":
                for (x, y, w, h) in self.cars_Cords:
                    cv2.circle(self.frame_RGB, (x+w//2, y+h//2),
                               ((w//2) + 5), color, BD)
            else:
                raise TypeError("Your Detection Frame Type Is Not Allowed!")

            cv2.imshow("Video Car Detector", self.frame_RGB)

            self.key = cv2.waitKey(1)
            if self.key == 81 or self.key == 113:
                break
        self.video.release()
        cv2.destroyAllWindows()
        print("\nRun Successfull.")
        print("\nAll Windows's Are Destroyed.")
