A. Simple code support training the model
1. Interact with the dataset download form kaggle
   + Labeling : labeling_dataset.py
   + Create test/vaidatate dataset that support for yolov8 :create_val_test.py
   + Check if dataset had duplicated image : check_duplicate.py
   + Change class on the label folder for approprate action : changing_label_class.py
   + Dowwnload the model immediately after trainning to avoid corruption : GG_colab_download_model.py
   + Interact data from roboflow to google class -> use some kind of method to perform tranning : Get_better_model_tf_minst_roboflow_complete.ipynb
   + Build the mass validate set with data capture from roboflow : Select_images_depend_on_class.py
2. Verify The result after tranning model :
   + validating_model.py



B.Model Training Sample :
1. Kaggle_training_traffic_sign_sharing_section.ipynb
- Download dataset from kaggle (kaggle dataset)
- Download the validate image set
- Show some images
- Use the default Yolov8 model to detect object in kaggle dataset 
- Train 1st model
- Use 1st model to detect object in kaggle dataset -> good 
- Use 1st model to detect object validate image set -> bad
- Use team model to detect object validate image set -> neutral , have a better result
-> The 1st model have been overfitted 


