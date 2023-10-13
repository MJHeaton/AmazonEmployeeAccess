##
## Analysis of the Amazon Employee Access Data
##

## Libraries I need
library(tidyverse)
library(tidymodels)
library(vroom)
library(stacks)
library(embed)
library(doParallel)

## Read in the Data
train <- vroom("./data/train.csv") %>%
  mutate(ACTION=factor(ACTION))
test <- vroom("./data/test.csv")

## Define the folds up front
folds <- vfold_cv(train, v=10)
metSet <- metric_set(roc_auc)

## Open up parallel processing
# all_cores <- detectCores(logical = FALSE)
# cl <- makePSOCKcluster(all_cores)
# registerDoParallel(cl)

## Define a Recipe and test it out
amazonRecipe <- recipe(ACTION~., data=train) %>%
  step_mutate_at(all_numeric_predictors(), fn=factor) %>%
  step_other(all_nominal_predictors, threshold = 0.0001) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome=vars(ACTION)) %>%
  step_normalize(all_numeric_predictors())
bakedTrain <- bake(prep(amazonRecipe), new_data=train)
dim(bakedTrain)

# targetEncodeRecipe <- recipe(ACTION~., data=train) %>%
#   step_mutate(ACTION=factor(ACTION)) %>%
#   step_mutate_at(all_numeric_predictors(), fn=factor) %>%
#   step_lencode_mixed(all_nominal_predictors(), outcome=vars(ACTION)) %>%
#   step_mutate(ACTION=as.numeric(ACTION)-1)
# bakedTarget <- bake(prep(targetEncodeRecipe), new_data=train)
# ggplot(data=bakedTarget, aes(x=RESOURCE, y=ACTION)) +
#   geom_point()


#########################
## Logistic Regression ##
#########################

## Define the model
logReg_model <- logistic_reg() %>%
  set_engine("glm")

## Set up workflow
logReg_wf <- workflow() %>%
  add_recipe(amazonRecipe) %>%
  add_model(logReg_model) %>%
  fit(data=train)

## Predict test set
logRegPreds <- logReg_wf %>%
  predict(new_data=test, type="prob") %>%
  bind_cols(test) %>%
  rename(ACTION=.pred_1) %>%
  select(id, ACTION)

## Write out the Predictions
vroom_write(x=logRegPreds, file="./LogRegPreds.csv", delim=",")

###################################
## Penalized Logistic Regression ##
###################################

## Define the model
penLogReg_model <- logistic_reg(mixture=tune(),
                             penalty=tune()) %>%
  set_engine("glmnet")

## Set up workflow
penLogReg_wf <- workflow() %>%
  add_recipe(amazonRecipe) %>%
  add_model(penLogReg_model)

## Use CV to tune the model
penLogReg_grid <- grid_regular(mixture(),
                               penalty(),
                               levels=10)
penReg_CV <- penLogReg_wf %>%
  tune_grid(resamples=folds,
            grid=penLogReg_grid,
            metrics=metSet)

penLogReg_wf <- penLogReg_wf %>%
  finalize_workflow(select_best(penReg_CV)) %>%
  fit(data=train)

## Predict test set
penLogRegPreds <- penLogReg_wf %>%
  predict(new_data=test, type="prob") %>%
  bind_cols(test) %>%
  rename(ACTION=.pred_1) %>%
  select(id, ACTION)

## Write out the Predictions
vroom_write(x=penLogRegPreds, file="./PenLogRegPreds.csv", delim=",")

###################
## Random Forest ##
###################

rf_model <- rand_forest(mtry=4, 
                        min_n=5,
                        trees=100) %>%
  set_engine("ranger") %>%
  set_mode("classification")
rf_wf <- workflow() %>%
  add_recipe(amazonRecipe) %>%
  add_model(rf_model) %>%
  fit(data=train)


#################
## Naive Bayes ##
#################

nb_model <- naive_Bayes(Laplace=tune(), smoothness=tune()) %>%
  set_mode("classification") %>%
  set_engine("naivebayes")

nb_wf <- workflow() %>%
  add_recipe(amazonRecipe) %>%
  add_model(nb_model)

nb_grid <- grid_regular(Laplace(),
                        smoothness(),
                        levels=10)

nb_cv <- nb_wf %>%
  tune_grid(resamples=folds,
            grid=nb_grid,
            metrics=metSet)

nb_wf <- nb_wf %>%
  finalize_workflow(select_best(nb_cv)) %>%
  fit(data=train)

nbPreds <- nb_wf %>%
  predict(new_data=test, type="prob") %>%
  bind_cols(test) %>%
  rename(ACTION=.pred_1) %>%
  select(id, ACTION)

## Write out the Predictions
vroom_write(x=nbPreds, file="./nbPreds.csv", delim=",")


###############################
## Turn off Parallel Cluster ##
###############################
stopCluster(cl)


