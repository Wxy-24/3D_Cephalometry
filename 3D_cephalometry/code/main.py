from models import*
import cv2
import numpy as np
from matplotlib import pyplot as plt
import skimage.io as io
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


#load data
data=np.load('/content/drive/My Drive/data35.npz')
coordinates=np.load('/content/drive/My Drive/z35ddl.npy')
name=np.load('/content/drive/My Drive/name21.npy')
hm=np.load('/content/drive/My Drive/3d_heatmap/3d_heatmap.npy')
bs=np.load('/content/drive/My Drive/bryant_sequence.npy')

X=data['ct']


# # optional rotation
# rotc = coordinates[:, [14, 18, 19], :].copy()
# bs = get_BryantSequence(rotc)
# xr=[]
# for i in range(35):
#   img=rot(bs[i],X[i])
#   xr.append(img)
# xr=np.array(xr)

#data generator
x=np.expand_dims(X,axis=4)
x_train, x_test, y_train, y_test =train_test_split(x,hm, test_size=0.085, random_state=666)
print(x_train.shape,y_train2.shape)
def myGenerator():
    while 1:
        for i in range(32):
          yield x_train[i*1:(i+1)*1], y_train2[i*1:(i+1)*1]

#training setting
model=load_3d_unet(pretrained_weights =None)
es =EarlyStopping(monitor='loss', min_delta=1e-5, patience=160, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2,patience=25, min_lr=1e-5)
point1 = ModelCheckpoint('/content/drive/My Drive/unet2.hdf5', monitor='val_loss',verbose=0, save_best_only=True)
history = model.fit_generator(myGenerator(),steps_per_epoch=32,epochs = 400,verbose=1,validation_data=(x_test,y_test),callbacks=[es,point1,reduce_lr])

# summarize history for loss
plt.plot(history.history['loss'][50:400])
plt.plot(history.history['val_loss'][50:400])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#prediction
result=model.predict(x_test)
result=result.transpose((4,0,1,2,3))
y_test=y_test.transpose((4,0,1,2,3))

#DISTANCE EVALUATION
num_landmarks=int(result.shape[0])
samples=int(result.shape[1])
total=num_landmarks*samples

def mean_euclidean_distance(gt,prediction):
  store=[]
  error=gt-prediction
  for i in range(total):
    distance=np.sqrt(np.sum(np.square(error[i])))
    store.append(distance)
  s=np.array(store)
  return np.mean(store)

gt=[]
for j in range(num_landmarks):
  for i in range(samples):
    landmarks=[int(np.where(y_test2[j,i,:,:]==np.max(y_test2[j,i,:,:]))[0]),int(np.where(y_test2[j,i,:,:]==np.max(y_test2[j,i,:,:]))[1]),int(np.where(y_test2[j,i,:,:]==np.max(y_test2[j,i,:,:]))[2])]
    gt.append(landmarks)
gt=np.array(gt)

#original
predict=[]
for j in range(num_landmarks):
  for i in range(samples):
    landmarks=[int(np.where(result[j,i,:,:]==np.max(result[j,i,:,:]))[0]),int(np.where(result[j,i,:,:]==np.max(result[j,i,:,:]))[1]),int(np.where(result[j,i,:,:]==np.max(result[j,i,:,:]))[2])]
    predict.append(landmarks)
predict=np.array(predict)
ssum=np.sum(np.square(gt-predict),axis=1)
for i in range(num_landmarks):
  print(name[i],':',round(0.33*(np.sqrt(ssum[samples*i])+np.sqrt(ssum[samples*i+1])+np.sqrt(ssum[samples*i+2])),2))
r=mean_euclidean_distance(gt,predict)
print("distance error for 3d unet is",str(round(r,2)))

