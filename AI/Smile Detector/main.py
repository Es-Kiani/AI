from Module.SmileDetector import LiveDetector

face_xml = "haarcascade_frontalface_default.xml"
smile_xml = "./haarcascade_smile.xml"
webcam_URL = "https://192.168.1.13:8569/video"

LiveDetector(face_xml, smile_xml).start(webcam_URL, 1.4, 4, 1.4, 5)
