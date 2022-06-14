# Pose_Detection

主要的執行地點（SVM、CNN train validation、online prediction）都位於 `main.ipynb`
你可以從`methods`裡面挑一個想要使用的方法去預測
然後mediapipe的train和validation在`mediapipe_training_testing.ipynb`

`label.py`和`capture.py`是用來製作dataset的，使用`capture.py`時要加上`--dataset_path`參數指定儲存資料夾