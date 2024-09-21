from ultralytics import YOLO 

model = YOLO('Player_Detection/models/2nd_approach/model_1.pt')

results = model.predict('/content/Juve_1.mp4',save=True)
print(results[0])
print('=====================================')
for box in results[0].boxes:
    print(box)