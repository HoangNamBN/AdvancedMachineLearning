## EE4208 Face Recognition

This project is an implementation of face recognition using PCA eigenfaces method.

```
ee4208_face_recognition
  ├── README.md
  ├── collect_faces.py
  ├── train.py
  ├── main.py
  ├── model
    ├── mean_face.csv
    ├── eigen_vectors.csv
    ├── eigen_faces.csv
    └── name_list.txt
  ├── dataset
      ├── 0.jpg
      ├── 1.jpg
      └── ...
  └── haarcascades
      └── haarcascade_frontalface_default.xml
```


### Requirements
* numpy
* opencv


### How to use
* clone the project
```
git clone https://github.com/jhchenjh/ee4208_face_recognition.git
```
* (optional) new data
```
python collect_faces.py
```
* train the face recognition model, model will be saved under *model* folder after training.
```
python train.py
```
* run face recognition
```
python main.py
```
