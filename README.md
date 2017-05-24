# Naive Bayes Classifier 
for the MNIST handwritten digit dataset

#### How to run the code 
```
python dataClassifier.py 
```
Tested to have an accuracy of 74.0% on validation set & 75.0% on test set on default settings.
By default smoothing_constant is 2, training set size is 1000 and test set size is 100.




```
python dataClassifier.py -k [smoothing_constant] -t [training_set_size] -x [test_set_size]
```

Tested to have an accuracy of 64.1% on validation set & 64.3% on test set 
(k = 2, t = 10000, x = 1000)

skeleton was taken from [source](http://www.cs.utexas.edu/~pstone/Courses/343spring12/assignments/classification/classification.html
