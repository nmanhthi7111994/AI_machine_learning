1st : !yolo task=detect mode=train model=/content/AI_machine_learning/Player_Detection/models/2nd_approach/model_1.pt data=/content/data.yaml epochs=100 imgsz=640 batch=16
2nd : !yolo task=detect mode=train model=/content/AI_machine_learning/Player_Detection/models/2nd_approach/model_1.pt data=/content/data.yaml epochs=100 imgsz=640 batch=16
3th : Use the custom training that focus to improve the ball : Add Custom Layers: Modify the YOLO model by adding custom detection layers tailored for detecting smaller objects. This can improve its ability to handle smaller objects like the ball.