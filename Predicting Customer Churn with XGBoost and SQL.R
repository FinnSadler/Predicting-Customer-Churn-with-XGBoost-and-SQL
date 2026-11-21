
#Predicting customer churn with the UCI ML Repository 'Online Retail II' dataset

#Load packages
library(tidyverse)
library(DBI)
library(duckdb)
library(readxl)
library(janitor)
library(broom)
library(recipes)
library(tidymodels)
library(xgboost)
library(probably)
options(scipen=999)

#Check file type and load in the dataset
path <- "D:\\Datasets\\online_retail_II.xlsx"
excel_format(path)

Retail <- clean_names(bind_rows(
  read_xlsx(path, sheet = 1),
  read_xlsx(path, sheet = 2)))

head(str(Retail))
tibble(Retail)

#Create a database file so the data can be accessed with SQL to save time
con <- dbConnect(duckdb(), "retail.duckdb")
dbWriteTable(con, "retail", Retail, overwrite = TRUE)
        
#Calculate number of customers with a missing id
dbGetQuery(
  con,
  "SELECT customer_id IS NULL AS missing_id, COUNT(*) AS n
  FROM retail
  GROUP BY 1"
) #243,000 rows missing customer_id, will need to be dropped


#Build distinct customer sample
features <- dbGetQuery(con, "
WITH before AS (
  SELECT * FROM retail
  WHERE customer_id IS NOT NULL
    AND invoice NOT LIKE 'C%'
    AND quantity > 0 AND price > 0
    AND invoice_date < TIMESTAMP '2011-09-01'
),
after AS (
  SELECT DISTINCT customer_id FROM retail
  WHERE customer_id IS NOT NULL
    AND invoice NOT LIKE 'C%'
    AND invoice_date >= TIMESTAMP '2011-09-01'
    AND invoice_date <  TIMESTAMP '2011-12-01'
),
core AS (
  SELECT customer_id,
         DATE_DIFF('day', MAX(invoice_date), TIMESTAMP '2011-09-01') AS recency_days,
         DATE_DIFF('day', MIN(invoice_date), TIMESTAMP '2011-09-01') AS tenure_days,
         COUNT(DISTINCT invoice)     AS frequency,
         SUM(quantity * price)       AS monetary,
         SUM(CASE WHEN invoice_date >= TIMESTAMP '2011-09-01' - INTERVAL 90 DAY
                  THEN quantity * price ELSE 0 END) AS spend_last_90d,
         arg_max(country, invoice_date) AS latest_country,
         COUNT(DISTINCT country)     AS n_countries,
         COUNT(DISTINCT stock_code)  AS n_products,
         AVG(quantity)               AS avg_qty_per_line,
         SUM(CASE WHEN description ILIKE '%CHRISTMAS%'
                  THEN quantity * price ELSE 0 END)
           / NULLIF(SUM(quantity * price), 0) AS christmas_share
  FROM before
  GROUP BY customer_id
)
SELECT c.*,
       CASE WHEN a.customer_id IS NULL THEN 1 ELSE 0 END AS churned
FROM core c
LEFT JOIN after a ON c.customer_id = a.customer_id
")

#Check variables
variable.names(features)

features <- features %>% 
  mutate(
  churned = factor(churned, levels = c(1, 0), labels = c("yes", "no"))
  )

#Split data
set.seed(1234)
featuresSplit <- initial_split(features, prop = 0.7, strata = churned)
featuresTrain <- training(featuresSplit)
featuresTest <- testing(featuresSplit)

#Inspect distribution of target variable in train and test sets
mean(featuresTrain$churned == "yes") #57%
mean(featuresTest$churned == "no") #~43% -- slightly higher churn rate, will need to stratify the split

featuresTrain %>% count(churned)

#Pre-processing
churnRecipe <- recipe(churned ~ ., data = featuresTrain) %>% #The formula for the model
  update_role(customer_id, new_role = "id") %>%
  step_zv(all_predictors()) %>% #Remove all columns with 0 variance, which is useless
  step_dummy(all_nominal_predictors(), one_hot = TRUE) #Dummy variable all categoricals
  #Remove customer id so model doesn't try to learn from it 

#Build XGBoost model
xgbSpec <- boost_tree(
  trees = 2000, #Number of boosting rounds (each tree corrects its predecessors' errors)
  tree_depth = tune(), #How complex each tree can be, deeper captures more, but overfits more easily
  learn_rate = 0.03, #How much each tree contributes, smaller needs more trees but generalises better
  min_n = tune(), #Minimum number of observations in a node before splitting, another brake on overfitting
  loss_reduction = tune(),
  sample_size = tune(),
  mtry = tune(),
  stop_iter = 50) %>%
  set_engine("xgboost", validation = 0.2, counts = FALSE) %>%
  set_mode("classification") #which package to use and which kind of problem

XgbTune <- workflow() %>%
  add_recipe(churnRecipe) %>%
  add_model(xgbSpec)

XgbParams <- extract_parameter_set_dials(XgbTune) %>%
  update(
    tree_depth = tree_depth(c(2, 8)),
    min_n = min_n(c(5, 60)),
    sample_size = sample_prop(c(0.5, 1)),
    mtry = mtry_prop(c(0.3, 1)),
    loss_reduction = loss_reduction(c(-10, 1), trans = transform_log10())
  )

foldsRep <- vfold_cv(featuresTrain, v = 5, repeats = 3, strata = churned)

XgbGrid <- grid_space_filling(XgbParams, size = 30)

XgbRes <- tune_grid(
  XgbTune,
  resamples = foldsRep,
  grid = XgbGrid,
  metrics = metric_set(roc_auc), #Optimise for a specific metric here
  control = control_grid(verbose = TRUE)
)

show_best(XgbRes, metric = "roc_auc", n = 5)

autoplot(XgbRes) + labs(title = "Space Filling Grid Search Results", y = "ROC AUC", x = "Hyperparameter Setting")

bestXGBoost <- select_best(XgbRes, metric = "roc_auc")

finalWorkflow <- finalize_workflow(XgbTune, bestXGBoost)

#Build a logistic regression baseline
folds <- vfold_cv(featuresTrain, v = 5, strata = churned) #stratified to keep churned proportion consistent across folds
churnMetrics <- metric_set(roc_auc, pr_auc, accuracy)

LRRecipe <- recipe(churned ~ ., data = featuresTrain) %>%
  update_role(customer_id, new_role = "id") %>%
  step_rm(latest_country) %>%
  step_zv(all_predictors()) %>%
  step_dummy(all_nominal_predictors())

LRWorkflow <- workflow() %>%
  add_recipe(LRRecipe) %>%
  add_model(logistic_reg() %>%
  set_engine("glm"))

Ctrl <- control_resamples(save_pred = TRUE)

LrCV <- fit_resamples(LRWorkflow, resamples = folds, metrics = metric_set(roc_auc), control = Ctrl)

collect_metrics(LrCV) #ROC_AUC = 0.800

#Test set
LrFinal <- last_fit(LRWorkflow, split = featuresSplit)
collect_metrics(LrFinal) #ROC_AUC = 0.765

finalRes <- last_fit(finalWorkflow, split = featuresSplit)
collect_metrics(finalRes) #ROC_AUC = 0.778

#Calibration curve
p <- collect_predictions(finalRes)

p %>%
  cal_plot_windowed(truth = churned, estimate = .pred_yes) +
  labs(title = "Calibration Curve for XGBoost Model",
       y = "Observed Frequency",
       x = "Predicted Probability")

