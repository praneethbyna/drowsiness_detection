#!/usr/bin/env python
from tkinter import Tk, Button
from subprocess import Popen
import os
import signal
import subprocess


root = Tk()
root.title("Driver Drowsiness Detector")
root.geometry("{}x{}".format(root.winfo_screenwidth(),root.winfo_screenheight()))

root.configure(background="light green") 
#root.attributes('-zoomed',True)
root.resizable(0,0)

def start():
    global process
    process = Popen("python /home/pi/Desktop/drowsiness-detection/drowsiness-detection/detect_drowsiness.py --shape-predictor shape_predictor_68_face_landmarks.dat --alarm alarm.wav",shell=True,preexec_fn=os.setsid)
    startbtn["state"]="disabled"

def stop():
    startbtn["state"]="normal"
    # Uncomment this if you want the process to terminate along with the window
    #process.terminate()
    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    #root.destroy()

def exitt():
    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    root.destroy()

def donothing():
    pass

startbtn=Button(root, text="Start", command=start,height=3, width=20)
startbtn.place(y=40,x=root.winfo_screenwidth()/2-100)

stopbtn=Button(root, text="Stop", command=stop,height=3, width=20)
stopbtn.place(y=140,x=root.winfo_screenwidth()/2-100)

exitbtn=Button(root, text="Exit The App", command=exitt,height=3, width=20)
exitbtn.place(y=240,x=root.winfo_screenwidth()/2-100)

#disabling the close button(x) on tkinter window
root.protocol('WM_DELETE_WINDOW',donothing) 
root.mainloop()
