pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}



library(Hmisc); library(caret); library(randomForest);

set.seed(68952)

# import
whole_data <- csv.get("pml-training.csv", na.strings=c("", "NA", "#DIV/0!"))
whole_test <- csv.get("pml-testing.csv", na.strings=c("", "NA", "#DIV/0!"))

Data <- whole_data
Test <- whole_test

# eliminate indices, timestamps, windows, variances for which i already have stddev
Data <- Data[,!grepl("X{1}|timestamp|window|var.(pi[ct]{2}h|yaw|roll)",names(Data))]
Test <- Test[,!grepl("X{1}|timestamp|window|var.(pi[ct]{2}h|yaw|roll)",names(Test))]


# proportion of NA values over all values, for each predictor
propNA = apply(is.na(Data),2,sum)/dim(Data)[1]
hist(propNA)

# eliminate sparse data
Data <- Data[,propNA<0.9]
Test <- Test[,propNA<0.9]




############################### second try: global random forest


objective = Data$classe
predictors = Data[, !grepl("user.name|classe",names(Data))]

eigth <- createDataPartition(y=objective,p=0.125,list=FALSE)
fourth <- createDataPartition(y=objective,p=0.25,list=FALSE)
half <- createDataPartition(y=objective,p=0.5,list=FALSE)

# check how many variables are useful to prediction (takes about 18 minutes)
system.time(result <- rfcv(predictors[half,], objective[half], step=0.9, cv.fold=4))
with(result, plot(n.var, error.cv, log="y", type="o", lwd=2))
with(result, error.cv * n.var)

MTRY = 9

# first try: random forest model (should take ~12 minutes)
system.time(rf_model <- randomForest(x=predictors, y=objective, replace=FALSE , mtry=MTRY,prox=TRUE,allowParallel=TRUE,importance=TRUE,do.trace=TRUE))



############################### second try: stacking user models

# divide by user
splitted <- split(Data, Data$user.name)

# get optimal number of predictors (should take)
objective_2 <- predictors_2 <- result_2 <- vector(mode="list", length=6)
for (i in 1:6){
  print(i)
  objective_2[[i]] <- splitted[[i]]$classe
  predictors_2[[i]] <- splitted[[i]][, !grepl("user.name|classe",names(Data))]      # eliminate user name, classe
  
  print(system.time(result_2[[i]] <- rfcv(predictors_2[[i]], objective_2[[i]], step=0.9, cv.fold=4)))
  with(result_2[[i]], print(error.cv * n.var))
}

# plot cross validation error for each user
colors<- rainbow(6)
with(result, plot(n.var, error.cv, log="x", type="o", lwd=3))
for (i in 1:6){with(result_2[[i]], lines(n.var, error.cv, col=colors[i], lwd=1.5, type="o"))}

# plot mean cross validation error
err_sum <- vector(mode="numeric", length=length(result_2[[1]]$error.cv))
for (i in 1:6){err_sum = err_sum + result_2[[i]]$error.cv}
lines(result$n.var, err_sum/6, lwd=1.5, type="o", pch="x")

MTRY = 9

# create random forests (should take ~2 minutes)
rf_model_2 <- vector(mode="list", length=6)
for(i in 1:6){
  message(system.time(rf_model_2[[i]] <- randomForest(x=predictors_2[[i]], y=objective_2[[i]], replace=FALSE , mtry=MTRY,prox=TRUE,allowParallel=TRUE,importance=TRUE,do.trace=TRUE)))
}



######################## PREDICTION #################

testing = Test[,!grepl("user.name|problem.id",names(Test))]
all(names(predictors)==names(testing))  # sanity check, all predictors are in testing set

####### first model
answers <- predict(rf_model, Test)

####### second model
answers_2 <- vector(mode= "integer", length = length(answers))
my_factors = c("adelmo", "carlitos", "charles", "eurico", "jeremy", "pedro")

answers_2 = factor(levels=c("A","B","C","D","E"))
for (i in 1:6){
  which_user = Test$user.name == my_factors[i]
  answers_2[which_user] <- predict(rf_model_2[[i]], Test[which_user,])
}

##### FINAL OUTPUT ####

all(answers == answers_2)
pml_write_files(answers)
