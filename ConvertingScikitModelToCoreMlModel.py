# -*- coding: utf-8 -*-
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
import coremltools


#Load the dataset : Iris (Flower Types)
iris = datasets.load_iris()

#Training model : with Logistic Regression Algorithm)
model = LogisticRegression()
model.fit(iris.data,iris.target)

#Make a prediction

print "Prediction with Scikit Iris Model"
# Sepal length , sepal width , petal length , petal width
print iris.target_names[model.predict([  [ 1.0 , 2.0, 0.0, 0.0]  ])]
joblib.dump(model,"iris.pkl")

# Exporting to coreml format
# Parameters: model , feature names ( virginica , setosa, ...)

coreml_model = coremltools.converters.sklearn.convert(model , iris.feature_names , "iris")
coreml_model.save("iris.mlmodel")


