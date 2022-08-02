import dlib
import numpy as np
import cv2
import pandas as pd
import os
import datetime as dt
import csv

def face_rec():
    tolerence = 0.43

    # face recognition model, the object maps human faces into 128D vectors

    facerec = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")

    # Computing the euclidean-distance between two 128D features
    def return_euclidean_distance(feature_1, feature_2):
        feature_1 = np.array(feature_1)
        feature_2 = np.array(feature_2)
        dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
        return dist

    attend = {} # To check for presence of a person
    verify = {} # Used for counting a particular person

    if os.path.exists("data/features_all.csv"):
        path_features_known_csv = "data/features_all.csv"
        person_list = os.listdir("data/data_faces_from_camera/")
        features_person_name = []
        csv_rd = pd.read_csv(path_features_known_csv, header=None)
        current_date_time = str(dt.datetime.now())
        today_date = current_date_time.split(' ')[0]

        # Making CSV file for storing the attendance data
        data = []
        for i in range(len(person_list)):
            data.append([person_list[i], "Absent", "-", "-"])
        df = pd.DataFrame(data, columns = ['Name', 'Attendance', 'In-Time', 'Out-Time'])
        # The array to save the features of faces in the database
        features_known_arr = []

        for i in range(csv_rd.shape[0]):
            features_someone_arr = []
            for j in range(0, len(csv_rd.iloc[i, :])):
                if j==0:

                    attend[str(csv_rd.iloc[i, :][j]).replace(".0" , "")] = 0
                    verify[str(csv_rd.iloc[i, :][j]).replace(".0" , "")] = 0
                    features_person_name.append(str(csv_rd.iloc[i, :][j]).replace(".0" , "") )
                    # print(csv_rd.iloc[i, :][j])
                    # exit()
                else:

                    features_someone_arr.append(csv_rd.iloc[i, :][j])

            features_known_arr.append(features_someone_arr)
        print(attend)
        # exit()
        print("Faces in Database: ", len(features_known_arr))
        # print(features_person_name)

        detector = dlib.get_frontal_face_detector() # Detects the faces in the frame

        # The detected faces are passed to predictor which checks for the feature-match for each face
        predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')

        threshold = 5
        cap = cv2.VideoCapture(0)

        curr_in_time = {}
        curr_out_time = {}
        # Capture the frames
        while cap.isOpened():
            flag, img_rd = cap.read()
            # img_gray = cv2.cvtColor(img_rd, cv2.COLOR_RGB2GRAY)
            img_gray = img_rd
            faces = detector(img_gray, 0)

            # font
            font = cv2.FONT_ITALIC

            # The list to save the positions and names of current faces captured
            pos_namelist = []
            name_namelist = []

            kk = cv2.waitKey(1)

            # press 'q' to exit
            if kk == ord('q'):
                break
            else:
                # when face detected
                if len(faces) != 0:
                    features_cap_arr = []
                    for i in range(len(faces)):
                        shape = predictor(img_rd, faces[i])
                        features_cap_arr.append(facerec.compute_face_descriptor(img_rd, shape))


                    for k in range(len(faces)):


                        # Set the default names of faces with "unknown"
                        name_namelist.append("unknown")

                        # the positions of faces captured
                        pos_namelist.append(tuple([faces[k].left(), int(faces[k].bottom() + (faces[k].bottom() - faces[k].top())/4)]))

                        # For every faces detected, compare the faces in the database
                        e_distance_list = []
                        for i in range(len(features_known_arr)):

                            if str(features_known_arr[i][0]) != '0.0':

                                e_distance_tmp = return_euclidean_distance(features_cap_arr[k], features_known_arr[i])

                                e_distance_list.append(e_distance_tmp)
                            else:
                                e_distance_list.append(999999999)
                        # Find the one with minimum e distance
                        # print(e_distance_list)
                        similar_person_num = e_distance_list.index(min(e_distance_list))


                        if min(e_distance_list) < tolerence:
                            # Here you can modify the names shown on the camera


                            name_namelist[k] = features_person_name[similar_person_num]

                            attend[name_namelist[k]] += 1
                            if attend[name_namelist[k]] > threshold and verify[name_namelist[k]] == 0:
                                curr_in_time[name_namelist[k]] = (str(dt.datetime.now()).split(' ')[1]).split('.')[0]
                                verify[name_namelist[k]] = 1
                                df.loc[df['Name']==name_namelist[k], ['Attendance']]= "Present"
                                df.loc[df['Name']==name_namelist[k], ['In-Time']] = curr_in_time[name_namelist[k]]
                            elif attend[name_namelist[k]] > threshold:
                                curr_out_time[name_namelist[k]] = (str(dt.datetime.now()).split(' ')[1]).split('.')[0]
                        else:
                            pass
                        # drawing rectangle boxes
                        for kk, d in enumerate(faces):
                            cv2.rectangle(img_rd, tuple([d.left(), d.top()]), tuple([d.right(), d.bottom()]), (0, 255, 255), 2)

                    # 6. write names under rectangle
                    for i in range(len(faces)):
                        cv2.putText(img_rd, name_namelist[i], pos_namelist[i], font, 0.8, (0, 255, 255), 1, cv2.LINE_AA)

            cv2.putText(img_rd, "Press 'q': Quit", (20, 450), font, 0.8, (84, 255, 159), 1, cv2.LINE_AA)
            cv2.putText(img_rd, "Face Recognition", (20, 40), font, 1, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(img_rd, "Faces: " + str(len(faces)), (20, 100), font, 1, (0, 0, 255), 1, cv2.LINE_AA)

            cv2.imshow("camera", img_rd)
        for key in curr_out_time:
            if int(curr_in_time[key].split(':')[1]) + 30 < int(curr_out_time[key].split(':')[1]) :
                df.loc[df['Name']==key,['Out-Time']] = curr_out_time[key]
            else:
                pass
        print(df)
        print(attend)
        df.to_csv("attendance_"+str(today_date)+".csv", index=False, header=True)



        if "cap" in globals():
            cap.release()
        cv2.destroyAllWindows()

    else:
        print('##### Warning #####', '\n')
        print("'features_all.py' not found!")
        print("Please run 'get_faces_from_camera.py' and 'features_extraction_to_csv.py' before 'face_reco_from_camera.py'", '\n')