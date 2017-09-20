from keras.layers import Input, Dense
from keras.models import Model
from keras import optimizers
import pandas as pt;
import numpy as np;
import math;
from matplotlib import pyplot as plt;

def getMax(tuple):
	index = 0;
	for i in tuple:
		if i == 1:
			return index;
		else:
			index+=1;

def findmax(tuple):
	tarr = np.zeros([1,3]);
	if (tuple[0] > tuple[1]):	
		if (tuple[0] > tuple[2]):		
			tarr[0][0] = 1;
		else:
			tarr[0][2] = 1;
		
	elif (tuple[1] > tuple[2]):
			tarr[0][1] = 1;
	else:
			tarr[0][2] = 1;
	
	return tarr;
#Load csv training data
iris_train_data = np.loadtxt('iris.csv', delimiter=',', dtype=np.float32);
iris_test_data = np.loadtxt('iris_test.csv', delimiter=',', dtype=np.float32);

#create variables for data times, data and target of both iris_Test and iris_train
iris_target = np.ones([150]);
iris_data   = np.ones([150,4]);
index = 0;

tiris_target = np.ones([30]);
tiris_data   = np.ones([30,4]);

#get csv values and add to data and target variables
for i in iris_train_data:
	iris_target[index] = math.ceil(i[4:5]);
	iris_data[index][0] = i[0];
	iris_data[index][1] = i[1];
	iris_data[index][2] = i[2];	
	iris_data[index][3] = i[3];
	index+=1;

index = 0;
for i in iris_test_data:
	tiris_target[index] = math.ceil(i[4:5]);
	tiris_data[index][0] = i[0];
	tiris_data[index][1] = i[1];
	tiris_data[index][2] = i[2];
	tiris_data[index][3] = i[3];
	index += 1;

index=0;

#Implement one-hot encoding for iris_Target data for test and target
iris_target_values = np.zeros([150, 3]);
for i in iris_target:
	target_row = np.zeros([1,3]);	
	target_row[0][int(i)] = 1;
	iris_target_values[index] = target_row;
	index+=1;

tiris_target_values = np.zeros([30,3]);
index=0;
for i in tiris_target:
	target_row = np.zeros([1,3]);
	target_row[0][int(i)] = 1;
	iris_target_values[index] = target_row;
	index += 1;

#normalize one-hot encoder matrix
iris_target_values/=10;
tiris_target_values /=10;

 #implement autoencoder using keras which runs on-top of tensorflow
iris_data   /= 10;
iris_target /= 10;

tiris_data /=10;
tiris_target /=10;
#Encoding Dimensions of Autoencoder - compressing
encoding_dim = 2;

#Input placeholder which has 4 input 'nodes'
input_iris = Input(shape=(4,));

#encoded layer of iris data, with a relu activation function
encoded = Dense(encoding_dim, activation='relu')(input_iris)

#decode layer attempts to reconstruct the input
decoded = Dense(4, activation='sigmoid')(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_iris, decoded);

#create a placeholder for encoded input
encoded_input = Input(shape=(encoding_dim,))

# this model maps an input to its encoded representation
encoder = Model(input_iris, encoded);

# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]

# create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))

#model configuration
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
autoencoder.compile(optimizer=sgd, loss='binary_crossentropy')

#train autoencoder model
autoencoder.fit(iris_data, iris_data,
                epochs=1000,                
                shuffle=True,
                batch_size=10,
                validation_data=(iris_data, iris_data))

#encoded layer from autoencoder 
encoded_iris = encoder.predict(iris_data);


#create deep net for classifing iris data using encoded layer
input_encoded = Input(shape=(2,));
layer_1 = Dense(15, activation="tanh")(input_encoded);
layer_2 = Dense(15, activation="tanh")(layer_1);
layer_3 = Dense(15, activation="tanh")(layer_2);
dnn_output = Dense(3,activation="sigmoid")(layer_3);

nn_iris = Model(input_encoded, dnn_output);
sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
nn_iris.compile(optimizer=sgd, loss="binary_crossentropy");

#train deep neural net model 
tencoder_iris = encoder.predict(tiris_data);
nn_iris.fit(encoded_iris, iris_target_values,
			epochs=1000,
			shuffle=True,
			batch_size=10,
			validation_data=(tencoder_iris, tiris_target_values));

#test data encoder
iris_pred = nn_iris.predict(tencoder_iris);


#encode iris_prediction one-hot 
dPrediction = np.zeros([30,3]);
index=0;
for i in iris_pred:
	tarr = findmax(iris_pred[index]);
	dPrediction[index] = tarr;
	index+=1;

print(tencoder_iris);
print(iris_pred);
print(dPrediction);
#visualize the data
colors=[];
tencoder_iris *= 10;


xs = [];
ys = [];

#add encoder to x and y axis
for i in tencoder_iris:
	xs.append(i[0]);
	ys.append(i[1]);

#convert classes to colors
for i in dPrediction:
	pclass = getMax(i);
	if (pclass == 0):
		color = 'R';
	if (pclass == 1):
		color = 'G';
	if (pclass == 2):
		color = 'B';
	colors.append(color);

#plot scatter	
plt.scatter(xs,ys, c=colors, alpha=.5);

#create and save model visualization
from keras.utils import plot_model
plot_model(nn_iris, to_file='model.png')


plt.show();

