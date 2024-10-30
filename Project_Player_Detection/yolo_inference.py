from ultralytics import YOLO 

model = YOLO(r'C:\Users\Admin\Documents\Github\AI_machine_learning\Project_Player_Detection\models\2nd_approach\8th_training_result.pt')

results = model.predict(r'C:\Users\Admin\Documents\Github\AI_machine_learning\Juve_2.mp4',save=True)
print(results[0])
print('=====================================')
for box in results[0].boxes:
    print(box)