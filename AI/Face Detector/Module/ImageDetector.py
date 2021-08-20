import cv2


class ImageDetector:
    def __init__(self, got_xml):
        self.trained_Data_xml = got_xml
        try:
            self.trained_Data = cv2.CascadeClassifier(
                cv2.data.haarcascades + self.trained_Data_xml)
        except:
            raise ValueError("Input xml File Could Not Open")

    def startDetection(self, img, type='r'):
        try:
            self.img_RGB = cv2.imread(img)
            self.img_GRY = cv2.cvtColor(self.img_RGB, cv2.COLOR_RGB2GRAY)
            self.face_Cords = self.trained_Data.detectMultiScale(self.img_GRY)
        except:
            raise ValueError("Could Not Find Image...!")

        if type == 'c':
            for (x, y, w, h) in self.face_Cords:
                cv2.circle(self.img_RGB,
                           (x+w//2, y+h//2), ((w//2) + 5), (255, 50, 200), 2)
        elif type == 'r':
            for (x, y, w, h) in self.face_Cords:
                cv2.rectangle(self.img_RGB,
                              (x, y), (x+w, y+h), (0, 255, 0), 2)
        else:
            raise ValueError("Your Detection Frame Type Is Not Allowed!")

        cv2.imshow("Image Face Detector", self.img_RGB)
        print("\nRun Successfull.")
        cv2.waitKey()
        print("\nAll Windows's Are Destroyed.")
