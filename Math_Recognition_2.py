# Program: Formula Reognition Using Singular Vector Decomposition
# Importing libraries
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import pytesseract

from tkinter import filedialog
from tkinter import *

# Giving path for training images
Path2 = './dataimage/'
files2 = os.listdir(Path2)
images = []

# Readling all the images and storing into an array
for name in files2:
    temp = cv2.imread(Path2+name)
    temp = cv2.cvtColor(temp,cv2.COLOR_BGR2GRAY)
    temp = cv2.resize(temp, (100,100), interpolation = cv2.INTER_AREA)
    images.append(temp.flatten())
    
    
# Substracting mean from all images for normalization    
images = np.array(images)
mu = np.mean(images)
images = images-mu
images = images.T
# print(images.shape)


# SVD function
u,s,v = np.linalg.svd(images, full_matrices=False)
#print (u.shape, s.shape , v.shape)


# Reading test image as an input, converting into 100*100 
# test = np.array(cv2.imread('./test_images/Area.png'))
root = Tk()
root.filename =  filedialog.askopenfilename(initialdir = "/C:/Users/dhava/OneDrive/Desktop/NM Project/test_images",title = "Select file",filetypes = (("PNG files","*.png"),("all files","*.*")))
test = np.array(cv2.imread(root.filename))
root.destroy()

test = cv2.cvtColor(test,cv2.COLOR_BGR2GRAY)
test = cv2.resize(test, (100,100), interpolation = cv2.INTER_AREA)

img = test.reshape(1, -1)

# Substracting mean
img = img-mu

img = img.T
# print(img[:][50])


# Dot product of test image and U matrix
test_x = np.empty(shape = (u.shape[0], u.shape[1]),  dtype=np.int8)
# print(test_x.shape)

for col in range(u.shape[1]):    
    test_x[:,col] = img[:,0] * u[:,col]

dot_test = np.array(test_x, dtype='int8').flatten()


# Dot product of all the images and U matrix
dot_train = np.empty(shape = (u.shape[0]*u.shape[1], u.shape[1]),  dtype=np.int8)
temp = np.empty(shape = (u.shape[0], u.shape[1]),  dtype=np.int8)


for i in range(images.shape[1]):
    for c in range(u.shape[1]):    
        temp[:,c] = images[:,i] * u[:,c]
        
    tempF = np.array(temp, dtype='int8').flatten()
    dot_train[:, i] = tempF[:]


# Substracting Two dot products
sub = np.empty(shape = (u.shape[0]*u.shape[1], u.shape[1]))

for col in range(u.shape[1]):
    sub[:,col] = dot_train[:,col] - dot_test[:]


# Finding norm of all the colums
answer = np.empty(shape=(u.shape[1],))

# Norms are for error calculation 
# We are detecting where minimum error is and finding its index
# finding same indexed filename from dataset and giving it to OCR (tesseract)

for c in range(sub.shape[1]):    
    answer[c] = np.linalg.norm(sub[:,c])  # Frobenius norm
#print(answer)

# Sorting answer array and retriving first element which will be minimum from all
temp_ans = np.empty(shape=(u.shape[1],))
temp=np.copy(answer)

temp.sort()
check = temp[0]
# print(check)


index=0

for i in range(answer.shape[0]):
    if check == answer[i]:
        
        index=i
 
        break

print("\n\n* * FORMULA DETECTION * * \n")
# Checking for corresponding image for minimum answer
folder_tr = '/dataimage/'
i = 0
print("\nNOTE : After ",index," Images We Got Final Detected File.....")


##############################################################################
import pytesseract

def OCR(fname):
    pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

    img = cv2.imread(f'C:\\Users\\dhava\\OneDrive\\Desktop\\NM Project\\test_images\\{fname}')

    text =  pytesseract.image_to_string(img)

    # print("\nImage Formula Value : ",text)
    return text
##############################################################################

#File name displays the actual filename with an extension like after successfully recognize, we got Area.png.
#OCR() is the tesseract function, which is used to display real image value on screen

#print(os.listdir(os.getcwd()+"/"+folder_tr))
for filename in os.listdir(os.getcwd()+"/"+folder_tr):
    
    if index == i:
        print("\nThe Final Predicted File Name : ",filename)
        # txt = filename
        # x = txt.split(".")
        print("\nPredicted Formula : ",OCR(filename))
        break
        
    else:
        i=i+1