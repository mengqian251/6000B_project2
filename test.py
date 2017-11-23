import numpy as np
import pandas as pd
from scipy.misc import imresize
from scipy.misc import imread
from keras.models import load_model

test = pd.read_table('val.txt', header=None, sep=' ', encoding='gb2312')
test.columns = ['path','label']


image_matrix = imread(test.iloc[0, :].path)
my_image = imresize(image_matrix, (224, 224))
height, width, depth = my_image.shape
num_test = len(test)
X_test = np.zeros((num_test, height, width, depth))

for i in range(num_test):
    image = imread(test.iloc[i, :].path)
    image = imresize(image_matrix, (224, 224))
    X_test[i, :, :, :] = image
    
my_model = load_model('my_model.h5')
my_predict = my_model.predict(X_test)

result = np.argmax(my_predict, axis=1)
print(result) 
np.savetxt("test_result.csv", result, delimiter=",")
