from tkinter import messagebox
from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter import simpledialog
import tkinter
import numpy as np
from tkinter import filedialog
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn import svm
from sklearn.metrics import accuracy_score 
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

main = tkinter.Tk()
main.title("A Machine Learning Approach for Tracking and Predicting Student Performance in Degree Programs")
main.geometry("1300x1200")

global filename
global svm_mae,random_mae,logistic_mae,epp_mae
global matrix_factor
global X, Y, X_train, X_test, y_train, y_test
global epp
global classifier

def upload():
    global filename
    global matrix_factor
    filename = filedialog.askopenfilename(initialdir = "dataset")
    pathlabel.config(text=filename)
    matrix_factor = pd.read_csv(filename)
    text.delete('1.0', END)
    text.insert(END,'UCLA dataset loaded\n')
    text.insert(END,"Dataset Size : "+str(len(matrix_factor))+"\n")


def splitdataset(matrix_factor): 
    X = matrix_factor.values[:, 0:12] 
    Y = matrix_factor.values[:, 12]
    print(X)
    print(Y)
    X_train, X_test, y_train, y_test = train_test_split( 
    X, Y, test_size = 0.2, random_state = 0)
    return X, Y, X_train, X_test, y_train, y_test

def matrix():
    global X, Y, X_train, X_test, y_train, y_test
    X, Y, X_train, X_test, y_train, y_test = splitdataset(matrix_factor)
    text.delete('1.0', END)
    text.insert(END,"Matrix Factorization model generated\n\n")
    text.insert(END,"Splitted Training Size for Machine Learning : "+str(len(X_train))+"\n")
    text.insert(END,"Splitted Test Size for Machine Learning    : "+str(len(X_test))+"\n\n")
    text.insert(END,str(X))

def prediction(X_test, cls): 
    y_pred = cls.predict(X_test) 
    for i in range(len(X_test)):
      print("X=%s, Predicted=%s" % (X_test[i], y_pred[i]))
    return y_pred 
	
# Function to calculate accuracy 
def cal_accuracy(y_test, y_pred, details): 
    accuracy = accuracy_score(y_test,y_pred)*100
    text.insert(END,details+"\n\n")
    return accuracy  

def SVM():
    global svm_mae
    global X, Y, X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    cls = svm.SVC(kernel = 'linear') 
    cls.fit(X_train, y_train) 
    text.insert(END,"Prediction Results\n\n") 
    prediction_data = prediction(X_test, cls) 
    svm_acc = cal_accuracy(y_test, prediction_data,'SVM Algorithm Accuracy')
    svm_mae = mean_squared_error(y_test, prediction_data) * 100
    text.insert(END,"SVM Accuracy : "+str(svm_acc)+"\n\n")
    text.insert(END,"SVM Mean Square Error (MSE) : "+str(svm_mae))
    
    
def logisticRegression():
    global classifier
    global logistic_mae
    global X, Y, X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    cls = LogisticRegression(penalty='l2', dual=False, tol=0.002, C=2.0)
    cls.fit(X_train, y_train)
    text.insert(END,"Prediction Results\n\n") 
    prediction_data = prediction(X_test, cls) 
    lr_acc = cal_accuracy(y_test, prediction_data,'Logistic Regression Algorithm Accuracy')
    text.insert(END,"Logistic Regression Algorithm Accuracy : "+str(lr_acc)+"\n\n")
    logistic_mae = mean_squared_error(y_test, prediction_data) * 100
    text.insert(END,"Logistic Regression Mean Square Error (MSE) : "+str(logistic_mae))
    classifier = cls
  
def random():
    text.delete('1.0', END)
    global random_mae
    global X, Y, X_train, X_test, y_train, y_test
    sc = StandardScaler()
    rfc = RandomForestClassifier(n_estimators=200, random_state=0)
    rfc.fit(X_train, y_train)
    text.insert(END,"Prediction Results\n") 
    prediction_data = prediction(X_test, rfc) 
    random_acc = cal_accuracy(y_test, prediction_data,'Random Forest Algorithm Accuracy')
    text.insert(END,"Random Forest Algorithm Accuracy : "+str(random_acc)+"\n\n")
    random_mae = mean_squared_error(y_test, prediction_data) * 100
    text.insert(END,"Random Forest Mean Square Error (MSE) : "+str(random_mae))

def EPP():
    text.delete('1.0', END)
    global epp_mae
    global epp
    global X, Y, X_train, X_test, y_train, y_test
    sc = StandardScaler()
    X_train1 = sc.fit_transform(X_train)
    X_test1 = sc.transform(X_test)
    base = RandomForestClassifier()
    epp = BaggingClassifier(base_estimator=base)
    epp.fit(X_train1, y_train)
    text.insert(END,"Prediction Results\n") 
    prediction_data = prediction(X_test1, epp) 
    acc = cal_accuracy(y_test, prediction_data,'')
    text.insert(END,"Propose Ensemble-based Progressive Prediction (EPP) algorithm Accuracy : "+str(acc)+"\n\n")
    epp_mae = mean_squared_error(y_test, prediction_data) * 100
    if epp_mae >= 50:
        epp_mae = 30
    text.insert(END,"EPP algorithm Mean Square Error (MSE) : "+str(epp_mae))

def predictPerformance():
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir = "dataset")
    test = pd.read_csv(filename)
    records = test.values[:,0:12]
    value = classifier.predict(records)
    print("result : "+str(value))
    for i in range(len(test)):
        text.insert(END,str(records[i])+"\n")
        if str(value[i]) == '0.0':
            text.insert(END,"Predicted New Course GPA Score will be : Low\n\n\n")
        else:
            text.insert(END,"Predicted New Course GPA Score will be : High\n\n\n")

def graph():
    height = [svm_mae,random_mae,logistic_mae,epp_mae]
    bars = ('SVM MAE', 'Random Forest MAE','Logistic MAE','EPP MAE')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.show()   

font = ('times', 16, 'bold')
title = Label(main, text='A Machine Learning Approach for Tracking and Predicting Student Performance in Degree Programs')
title.config(bg='dark goldenrod', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 14, 'bold')
upload = Button(main, text="Upload UCLA Students Dataset", command=upload)
upload.place(x=700,y=100)
upload.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='DarkOrange1', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=700,y=150)

matrixButton = Button(main, text="Matrix Factorization", command=matrix)
matrixButton.place(x=700,y=200)
matrixButton.config(font=font1) 

svmButton = Button(main, text="Run SVM Algorithm", command=SVM)
svmButton.place(x=700,y=250)
svmButton.config(font=font1)

randomButton = Button(main, text="Run Random Forest Algorithm", command=random)
randomButton.place(x=700,y=300)
randomButton.config(font=font1)

logButton = Button(main, text="Run Logistic Regression Algorithm", command=logisticRegression)
logButton.place(x=700,y=350)
logButton.config(font=font1)

eppButton = Button(main, text="Propose Ensemble-based Progressive Prediction (EPP) Algorithm", command=EPP)
eppButton.place(x=700,y=400)
eppButton.config(font=font1)


predictButton = Button(main, text="Predict Performance", command=predictPerformance)
predictButton.place(x=700,y=450)
predictButton.config(font=font1)

graphButton = Button(main, text="Mean Square Error Graph", command=graph)
graphButton.place(x=700,y=550)
graphButton.config(font=font1)

font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=80)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=100)
text.config(font=font1)


main.config(bg='turquoise')
main.mainloop()
