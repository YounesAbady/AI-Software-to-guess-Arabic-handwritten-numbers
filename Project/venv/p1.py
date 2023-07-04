import numpy as np
import matplotlib.pyplot as pt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from PIL import Image,ImageOps,ImageGrab
import cv2 as cv
import tkinter as tk
from tkinter import *
from win32 import win32gui
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import PIL.Image
#######################################################################
#reading Files as data and labels for each training,testing
#only took 20k from 60 because of my pc memory limit


x=pd.read_csv("D:\FCAI\Year 3\Semster 1\AI\GP\csvTrainImages 60k x 784.csv")
y=pd.read_csv("D:\FCAI\Year 3\Semster 1\AI\GP\csvTrainLabel 60k x 1.csv")
testData=pd.read_csv("D:\FCAI\Year 3\Semster 1\AI\GP\csvTestImages 10k x 784.csv")
testLabel=pd.read_csv("D:\FCAI\Year 3\Semster 1\AI\GP\csvTestLabel 10k x 1.csv")
trainData=x[:20000]
trainLabel=y[:20000]




######################################################################

######################################################################
#for fitting  DT and Getting accuarcy and printing img with predicting

# clf=DecisionTreeClassifier()
# clf.fit(trainData,trainLabel)
# p=clf.predict(testData)
# print("Accuarcy =",metrics.accuracy_score(testLabel,p)*100,'%')
# for x in range(5,10):
#     y_predicted = clf.predict((testData.iloc[x].values).reshape(1, -1))
#     label = y_predicted
#     pixel = testData.iloc[x]
#     pixel = np.array(pixel, dtype='uint8')
#     pixel = pixel.reshape((28, 28))
#     pt.title('its probabily = {label}'.format(label=label))
#     pixel=np.transpose(pixel)
#     pt.imshow(pixel, cmap='gray')
#     pt.show()

#######################################################################


#######################################################################
#for fitting RF and getting accuarcy and printing img with predicting

#
# x_train,x_test,y_train,y_test=train_test_split(trainData,trainLabel)
# model =RandomForestClassifier()
# model.fit(x_train,y_train)
# print("Accuarcy = ",model.score(x_test,y_test)*100,'%')
#
# for x in range(0,5):
#     y_predicted = model.predict((x_test.iloc[x].values).reshape(1, -1))
#     label = y_predicted
#     pixel = x_test.iloc[x]
#     y_pred = model.predict(x_test)
#     acc = accuracy_score(y_pred, y_test) * 100
#     pixel = np.array(pixel, dtype='uint8')
#     pixel = pixel.reshape((28, 28))
#     pt.title('its probabily = {label} with acc = {acc} %'.format(label=label, acc=acc))
#     pixel = np.transpose(pixel)
#     pt.imshow(pixel, cmap='gray')
#     pt.show()


#######################################################################
######################################################################
#take input from user
# path="D:/FCAI/Year 3/Semster 1/AI/GP/HandWrittenTest/"
# for x in range (1,10):
#     img=cv.imread(path+"{x}.png".format(x=x))[:,:,0]
#     img=np.invert(np.array([img]))
#     img = np.transpose(img)
#     label=model.predict(img.reshape(1, -1))
#     img = np.transpose(img)
#     pt.title('its probabily = {label} '.format(label=label))
#     pt.imshow(img[0],cmap='gray')
#     pt.show()


######################################################################

#######################################################################
#making gui that takes handwritten number(draw) then predict it

#
# def predict_digit(img):
#     img=img.resize((28,28))
#     img=img.convert('L')
#     img=np.invert(np.array(img))
#     img=img.reshape(1,-1)
#     res=model.predict(img)
#     return res
# class gui(tk.Tk):
#     def __init__(self):
#         tk.Tk.__init__(self)
#         tk.Tk.title="Digit Recogntion"
#         self.x=self.y=0
#         self.canvas=tk.Canvas(self,width=300,height=300,bg="white",cursor="cross")
#         self.label=tk.Label(self,text="Draw",font=("Helvetiva",48))
#         self.classify_btn=tk.Button(self,text="Recognise",command=self.classify_handwriting)
#         self.button_clear=tk.Button(self,text="Clear",command=self.clear_all)
#
#         self.canvas.grid(row=0,column=0,pady=2,sticky=W,)
#         self.label.grid(row=0,column=1,pady=2,padx=2)
#         self.classify_btn.grid(row=1,column=1,pady=2,padx=2)
#         self.button_clear.grid(row=1,column=0,pady=2)
#         self.canvas.bind("<B1-Motion>",self.draw_lines)
#
#     def classify_handwriting(self):
#         HWND=self.canvas.winfo_id()
#         rect=win32gui.GetWindowRect(HWND)
#         a,b,c,d=rect
#         rect=(a+4,b+4,c-4,d-4)
#         im=ImageGrab.grab(rect)
#         digit=predict_digit(im)
#         self.label.configure(text=str(digit))
#
#     def draw_lines(self,event):
#         self.x=event.x
#         self.y=event.y
#         r=8
#         self.canvas.create_oval(self.x-r,self.y-r,self.x+r,self.y+r,fill='black')
#     def clear_all(self):
#         self.canvas.delete("all")
#         self.label.configure(text="Draw")
#
#
# start=gui()
# mainloop()


#######################################################################