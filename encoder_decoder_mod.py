from keras.layers import Input, Dense
from keras.layers.core import Reshape
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.layers.convolutional import MaxPooling2D,UpSampling2D,Conv2DTranspose
from keras.layers.convolutional import Convolution2D as Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers import merge
from keras.optimizers import Adadelta, RMSprop
import os
import os.path
import numpy as np
from PIL import Image
from numpy import * 
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split

path1 = "/local/user/Desktop/Palm_casia/all"
batch_size = 128

######### convolutional encoder model

rms=RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.001)

def model1():

    input_img = Input(shape=(64,64,1))
    conv1 = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
      
    conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
       
    conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    
    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)
   
    
    encoder = Model(input_img, conv5)
    #autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    encoder.compile(loss='mean_squared_error', optimizer=rms)
    return encoder

def model2():
	input_img = Input(shape=(8,8,64))
	up6 = UpSampling2D((2,2))(input_img)
	conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(up6)
	conv6 = BatchNormalization()(conv6)
	conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv6)
	conv6 = BatchNormalization()(conv6)
	up7 = UpSampling2D((2,2))(conv6)
	conv7 = Conv2D(64, (3, 3), activation='relu', padding='same')(up7)
	conv7 = BatchNormalization()(conv7)
	conv7 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv7)
	conv7 = BatchNormalization()(conv7)
	up8 = UpSampling2D((2,2))(conv7)
	conv8 = Conv2D(32, (3, 3), activation='relu', padding='same')(up8)
	conv8 = BatchNormalization()(conv8)
	conv8 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv8)
	conv8 = BatchNormalization()(conv8)
	conv9 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv8)
	conv9 = BatchNormalization()(conv9)
	conv9 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv9)
	conv9 = BatchNormalization()(conv9)
	decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(conv9)
	
	decoder = Model(input_img, decoded)
	decoder.compile(loss='mean_squared_error', optimizer=rms)
	return decoder


encoder = model1()
# this model maps an input to its reconstruction
decoder = model2()


model_combined = Model(input = encoder.input,output = decoder(encoder.output))
ada = Adadelta(lr = 5.0, rho = 0.95, epsilon = 1e-08, decay = 0.001)
rms = RMSprop(lr = 0.001, rho = 0.9, epsilon = 1e-08, decay = 0.001)
model_combined.compile(loss = 'mean_squared_error', optimizer = rms)
  
list_files = os.listdir(path1)
#print(list_files)
original_matrix=[]
transformed_matrix=[]
for image in list_files:
	img = array(Image.open(path1 + "/" + image))
	img = np.expand_dims(img,axis=3)
	original_matrix.append(img)
transformed_matrix = original_matrix
data,Label = shuffle(original_matrix,transformed_matrix, random_state=2)
X_train, X_test, y_train, y_test = train_test_split(data, Label, test_size=0.3, random_state=2)
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

x_train = X_train.astype('float32') / 255.
x_test = X_test.astype('float32') / 255.
y_train = y_train.astype('float32') / 255.
y_test = y_test.astype('float32') / 255.


########### train the encoder decoder network
#autoencoder.load_weights('Trans_1_to_2.h5')
for epoch in range(100):
	train_X,train_Y = shuffle(x_train,y_train)
	print ("Epoch is: %d\n" % epoch)
	print ("Number of batches: %d\n" % int(len(train_X)/batch_size))
	num_batches = int(len(train_X)/batch_size)
	for batch in range(num_batches):
		batch_train_X = train_X[batch*batch_size:min((batch+1)*batch_size,len(train_X))]
		batch_train_Y = train_Y[batch*batch_size:min((batch+1)*batch_size,len(train_Y))]
		loss = model_combined.train_on_batch(batch_train_X,batch_train_Y)
		print ('epoch_num: %d batch_num: %d loss: %f\n' % (epoch,batch,loss))
	encoder.save("encoderModel.hdf5") 
	model_combined.save_weights("ear_model.h5")
	x_test,y_test = shuffle(x_test,y_test)
	decoded_imgs = model_combined.predict(x_test[:15])

	img = Image.fromarray(y_test[1].reshape(64,64)*255).convert('RGB')
	img.save('Trans_1_to_2/'+str(epoch)+'ytest.jpg')
	img = Image.fromarray(decoded_imgs[1].reshape(64,64)*255).convert('RGB')
	img.save('Trans_1_to_2/'+str(epoch)+'dtest.jpg')
