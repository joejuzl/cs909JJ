cs909JJ
=======

Coursework for the module CS909 (Datamining)


The python code in code.py is used to extract pass the reuters data, extract features and convert to CSV.
The code contains a number of self explainatory functions.
Most of the functions automatically save the data once they have processed it using a pickle function.
Setting the flag to true can be used to load the data rather than re-processing it. 
I have included the pickled feature sets but not the pre-processed corpus as it was too large. 
Running the code as it is will parse and process all the data, extract the LDA feature set and write it to CSV.
I have included the CSV files for all the feature sets as it takes a long time to process all the data.

Once the CSV files have been made (or using the included ones) R can be used to classify and cluster the data.
Make sure the following R packages are loaded:
cluster
class
e1071
fpc
mclust
randomForest

The files classifier* contain the R functions I made for clasifying and analysing the data.

The following commands show how to do everything in R:


#load my classifier functions. 
source("classifier1")
source("classifier2")
source("classifierF")

#load the test and train feature sets
uniTrain = read.csv("uniTrain.csv")
uniTest = read.csv("uniTest.csv")
biTrain = read.csv("biTrain.csv")
biTest = read.csv("biTest.csv")
ldaTrain = read.csv("ldaTrain.csv")
ldaTest = read.csv("ldaTest.csv")
tfidfTrain = read.csv("tfidfTrain.csv")
tfidfTest = read.csv("tfidfTest.csv")

#combine LDA and unigrams
combTrain = merge(ldaTrain[1:length(ldaTrain)-1],uniTrain,by=c("ID"))
combTest = merge(ldaTest[1:length(ldaTest)-1],uniTest,by=c("ID"))

#train random forest on the training data and run on the test data.
c2(combTrain,combTest,3)

#run naive bayes on the training data with 10 fold cross validation
#the second parameter is the number of folds. The third dictates the algorithm:
#1: naive bayes. 2: SVM. 3: random forest.
c1(combTrain,10,1)

#create the data for clustering
cdata = combTrain[,2:length(combTrain)-1]
classes = combTrain[,length(combTrain)]

#run all the clustering algorithms and display the confusion matrix with the original classes
k = kmeans(cdata,10)
table(classes,k$cluster)

d = dbscan(cdata,3)
table(classes,d$cluster)

m = Mclust(cdata)
table(classes,m$classification)

p = pam(cdata,10)
table(classes,p$cluster)

