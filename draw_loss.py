import matplotlib.pyplot as plt
import numpy as np
###############################################
# Load data
train_pts = []
val_pts = []
epoch_pts = []

f_train = open("train_loss.txt", "rb")
for line in f_train:
    train_pts.append(float(line.strip()))
f_train.close()

f_val = open("val_loss.txt", "rb")
for line in f_val:
    val_pts.append(float(line.strip()))
f_val.close()
##############################################
# Draw
x = np.linspace(0,len(train_pts),len(train_pts))
plt.plot(x,train_pts,'ro-')
plt.plot(x,val_pts,'bo-')
# plt.axis([0,len(train_pts), 0.02, 0.08])
plt.show()