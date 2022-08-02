from tkinter import *
from tkinter import font as tkFont
from get_faces_from_camera import get_face
from features_extraction_to_csv import feature_extract
from face_reco_from_camera import face_rec

root=Tk()
frame = Frame(root)
frame.pack()

myFont = tkFont.Font(size=40,family="Arial")

label = Label(root, text = "Face Recognition based Attendance System", fg = "Black", font = "Arial")
label.pack()

btn=Button(root, text="Face Registration", fg='white',bg="Red",activebackground="yellow",height=2,width=200,command=get_face)
btn['font'] = myFont
btn.pack()


btn1=Button(root, text="Feature Extraction", fg='white',bg="Green",activebackground="yellow",height=2,width=200,command=feature_extract)
btn1['font'] = myFont
btn1.pack()

btn2=Button(root, text="Take Attendence", fg='white',bg="blue",activebackground="yellow",height=2,width=200,command=face_rec)
btn2['font'] = myFont
btn2.pack()

root.title('Smart Attendance System')
root.geometry("900x500+10+20")
root.mainloop()

