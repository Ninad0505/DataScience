from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

digits = load_digits()

#print('Image Data Set', digits.data.shape)
#print('Label data shape', digits.target.shape)

#plt.figure(figsize=(20,4))
#for index, (image, label) in enumerate(zip(digits.data[0:5], digits.target[0:5])):
#    plt.subplot(1,5, index+1)
#   plt.imshow(np.reshape(image, (8,8)), cmap = plt.cm.gray)
#   plt.title('Training: %i\n' %label, fontsize = 20)
#   plt.show()
#print(digits.keys())

x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.23, random_state=2)

#print(x_train.shape)
from sklearn.linear_model import LogisticRegression
logisticregr = LogisticRegression()
logisticregr.fit(x_train, y_train)

#print(logisticregr.predict(x_test[0].reshape(1,-1)))
#print(logisticregr.predict(x_test[0:10]))

score = logisticregr.score(x_test, y_test)
#print(score)

predictions = logisticregr.predict(x_test)

cm = metrics.confusion_matrix(y_test, predictions)
#print(cm)

#heat map
plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt= '.3f', linewidths=0.5, square = True, cmap = 'Blues_r' );
plt.ylabel('Actual-Label')
plt.xlabel('Predicted-Label')
all_sample_title = 'Accuracy Score : {0}'.format(score)
plt.title(all_sample_title, size= 15)
plt.show()