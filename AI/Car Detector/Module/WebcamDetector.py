import cv2


class LiveDetector:
    def __init__(self, got_xml):
        self.trained_Data_xml = got_xml
        try:
            self.car_Detector = cv2.CascadeClassifier(self.trained_Data_xml)
        except:
            raise ValueError("Could Not Open the Input XML File")

    def startDetection(self,webcam_URL=0, type='r'):
        self.webcam = cv2.VideoCapture(webcam_URL)
        while 1:
            self.isReadSuccessfull, self.frame_RGB = self.webcam.read()
            if (self.isReadSuccessfull == False):
                raise ConnectionError("Could Not Find Webcam...!")

            self.frame_GRY = cv2.cvtColor(self.frame_RGB, cv2.COLOR_BGR2GRAY)
            self.cars_Cords = self.car_Detector.detectMultiScale(
                self.frame_GRY)

            if type == 'r':
                for (x, y, w, h) in self.cars_Cords:
                    cv2.rectangle(self.frame_RGB, (x, y),
                                  (x+w, y+h), (0, 255, 0), 2)
            elif type == "c":
                for (x, y, w, h) in self.cars_Cords:
                    cv2.circle(self.frame_RGB, (x+w//2, y+h//2),
                               ((w//2) + 5), (255, 50, 200), 2)
            else:
                raise TypeError("Your Detection Frame Type Is Not Allowed!")

            cv2.imshow("Live Webcam Car Detector", self.frame_RGB)

            self.key = cv2.waitKey(1)
            if self.key == 81 or self.key == 113:
                break
        self.webcam.release()
        cv2.destroyAllWindows()
        print("\nRun Successfull.")
        print("\nAll Windows's Are Destroyed.")
