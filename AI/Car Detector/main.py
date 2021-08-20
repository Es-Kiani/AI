from Module.VideoDetector import VideoDetector

VideoFile1 = "./Movie/car-detection.mp4"
VideoFile2 = "./Movie/dataset_video1.avi"
VideoFile3 = "./Movie/dataset_video2.avi"
VideoFile4 = "./Movie/person-bicycle-car-detection.mp4"

car_xml = "./cars.xml"

VideoDetector = VideoDetector(car_xml)
VideoDetector.startDetection(VideoFile2, BD=1)
