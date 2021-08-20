import cv2


class ImageDetector:
    def __init__(self, got_xml):
        self.trained_Data_xml = got_xml
        try:
            self.car_Detector = cv2.CascadeClassifier(self.trained_Data_xml)
        except:
            raise ValueError("Input xml File Could Not Open")

    def startDetection(self, img, type='r'):
        try:
            self.img_RGB = cv2.imread(img)
        except:
            raise ValueError("Could Not Find Image...!")
        
        self.img_GRY = cv2.cvtColor(self.img_RGB, cv2.COLOR_RGB2GRAY)
        self.cars_Cords = self.car_Detector.detectMultiScale(self.img_GRY)
        
        if type == 'c':
            for (x, y, w, h) in self.cars_Cords:
                cv2.circle(self.img_RGB, (x+w//2, y+h//2),
                           ((w//2) + 5), (255, 50, 200), 2)
        elif type == 'r':
            for (x, y, w, h) in self.cars_Cords:
                cv2.rectangle(self.img_RGB, (x, y),
                              (x+w, y+h), (0, 255, 0), 2)
        else:
            raise ValueError("Your Detection Frame Type Is Not Allowed!")
        
        cv2.imshow("Image Car Detector", self.img_RGB)
        print("\nRun Successfull.")
        cv2.waitKey()
        print("\nAll Windows's Are Destroyed.")
