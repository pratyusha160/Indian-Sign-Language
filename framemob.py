import cv2
import os

input_dir = "mobiledataset"
output_dir = "mobframes"
os.makedirs(output_dir, exist_ok=True)

for sentence in os.listdir(input_dir):
    sentence_path = os.path.join(input_dir, sentence)
    frame_path = os.path.join(output_dir, sentence)
    os.makedirs(frame_path, exist_ok=True)
    
    for video_file in os.listdir(sentence_path):
        video_path = os.path.join(sentence_path, video_file)
        cap = cv2.VideoCapture(video_path)
        count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_file = os.path.join(frame_path, f"{video_file[:-4]}_{count}.jpg")
            cv2.imwrite(frame_file, frame)
            count += 1
        
        cap.release()