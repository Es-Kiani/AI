from Module.ImageDetector import ImageDetector
from Module.VideoDetector import VideoDetector
from Module.WebcamDetector import LiveDetector

FrontFace_xml = "haarcascade_frontalface_default.xml"

img1 = "./Photo/Face1_Woman_RGB_OnePerson.jpg"
img2 = "./Photo/Face2_Both_RGB_8Person_Grimm.png"
videoFile = "./Movie/head-pose-face-detection-female.mp4"

# ImgDetector1 = ImageDetector(FrontFace_xml)
# ImgDetector1.startDetection(img1)
# # ImgDetector1.startDetection('c')

# ImgDetector2 = ImageDetector(FrontFace_xml)
# ImgDetector2.startDetection(img2)
# # ImgDetector2.startDetection('c')

# VideoDetector = VideoDetector(FrontFace_xml)
# VideoDetector.startDetection(videoFile)
# # VideoDetector.startDetection(videoFile, 'c')

webcam_URL = "https://192.168.1.13:8569/video"

LiveDetector = LiveDetector(FrontFace_xml)
LiveDetector.startDetection()
# LiveDetector.startDetection(type = 'c')

