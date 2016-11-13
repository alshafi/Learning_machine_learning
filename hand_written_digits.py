import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn import svm

digits = datasets.load_digits()

classifier = svm.SVC(gamma=0.0001, C=100)
x, y = digits.data[:-10], digits.target[:-10]
# training
classifier.fit(x, y)
# manual testing
print 'Prediction:', classifier.predict(digits.data[-4])
plt.imshow(digits.images[-4], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()
