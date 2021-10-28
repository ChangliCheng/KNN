#Dessislava A. Pachamanova
#kNN Example with Categorical Target Variable: Lawn Mower Ownership
#Steps for using kNN:
#1. Read in data; split into training and test sets
#2. Scale predictors (NOT target variable)
#3. Find nearest neighbors based on scaled distances
#4. Predict category of new observation based on nearest neighbors

if (!require("caret")) {
  install.packages("caret")
  library("caret")
}

if (!require("DMwR")) {
  install.packages("DMwR")
  library("DMwR")
}

#Import housing_cat.csv file
myData <- read.csv("housing_cat.csv", stringsAsFactors = TRUE)
anyNA(myData)
myData$CAT.MEDV <- as.factor(myData$CAT.MEDV)
#Split data into a random 75% training set and 25% test set
trainSetSize <- floor(0.75 * nrow(myData))   
set.seed(50)                       
trainInd <- sample(seq_len(nrow(myData)), size = trainSetSize) 
myDataTrain <- myData[trainInd, ]               
myDataTest <- myData[-trainInd, ] 
dim(myDataTrain)
dim(myDataTest)

myDataTrainScaled <- myDataTrain
myDataTestScaled <- myDataTest
myDataScaled <- myData


#use preProcess from the caret package to normalize Income and Lot_Size
normValues <- preProcess(myDataTrain[,1:10], method = c("center", "scale"))
myDataTrainScaled[,1:10] <- predict(normValues, myDataTrain[, 1:10])
myDataTestScaled[,1:10] <- predict(normValues, myDataTest[, 1:10])
myDataScaled[,1:10] <- predict(normValues, myData[, 1:10])


#Compute kNN
###########################
if (!require("FNN")) {
  install.packages("FNN")
  library("FNN")
}



#predict values for test set (not just a single observation)
predTestClass <- knn(train = myDataTrainScaled[,1:10], test = myDataTestScaled[,1:10], cl = myDataTrainScaled[, 11], k = 7)

#Compute confusion matrix for prediction model
################################################
actualTestClass <- myDataTestScaled$CAT.MEDV

#Calculate all relevant statistics for confusion matrix
#############################################
confMxAll <- confusionMatrix(predTestClass, actualTestClass, positive="1")
confMxAll
#actual confusion matrix
confMx <- confMxAll$table
confMx

#calculate total accuracy
totAcc <- confMxAll$overall[1]
totAcc

newObsScaled <- newObs
newObsScaled <- predict(normValues, newObs)
#new observation (household) to illustrate method
newObs <- data.frame(CRIM = 0.4000, INDUS = 7.00, NOX = 0.400, CHAS = 0, 
                     RM = 4.20, AGE = 50, 
                     DIS = 5.45, RAD = 3, TAX = 300, LSTAT = 4.00)

newObsScaled <- predict(normValues, newObs)
#predict a single observation (the new one) using k = 7 neighbors
#cl is the factor of the true classifications of the training set
predNewObs <- knn(train = myDataTrainScaled[,1:10], test = newObsScaled, cl = myDataTrainScaled[, 11], k = 7)
#find the k nearest neighbors and the (scaled) distances from this observation to its nearest k neighbors
row.names(myDataTrain)[attr(predNewObs,"predNewObs.index")]
predNewObs
#Decide how many neighbors to use based on which number results in highest model accuracy 
#################################
#Initiate a data frame (table) with first column = k and second column = Error
#The number of neighbors and the associated model error will be stored there
errTable <- data.frame(k = c(1:18),  err = rep(0, 18))
actualTestClass <- myDataTestScaled$CAT.MEDV

#fill out table
minErr <- 1
for (i in 1:18) {
  predTestClass <- knn(myDataTrainScaled[,1:10], myDataTestScaled[,1:10], cl = myDataTrainScaled[,11], k = i)
  predTestClass
  confMxAll <- confusionMatrix(predTestClass, myDataTestScaled$CAT.MEDV)
  totAcc <- confMxAll$overall[1]  
  i
  errTable[i, 2] <- 1 - totAcc
  errTable[i,2]
  if (errTable[i, 2] < minErr){ 
    minErr <- errTable[i, 2]
    bestK <- i
    bestConfMx <- t(confMxAll$table) 
    bestPredTestClass <- predTestClass
  } 
}
bestK
bestConfMx

#Save and export the best model and its predictions. A file will be created in the ROutput directory.
dfToExport <- data.frame(myDataTest,bestPredTestClass)
write.csv(dfToExport, file = "../ROutput/predictedOwnKNN.csv")