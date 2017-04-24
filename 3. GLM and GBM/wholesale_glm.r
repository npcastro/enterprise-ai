library(h2o)
h2o.init()

######## Data exploration ########

# Read dataset into dataframe
data<-h2o.importFile("/Users/npcastro/workspace/enterprise-ai/3. GLM and GBM/Wholesale customers data.csv")

# Check first rows of the dataset
h2o.head(data)

# See further details
h2o.describe(data)

# Assign appropiate types to variables

data["Channel"] = h2o.asfactor(data["Channel"])
data["Region"] = h2o.asfactor(data["Region"])

# Validate
h2o.describe(data)

# Analize data summary
h2o.summary(data)

# Group by available categories
h2o.group_by(data, by=c("Channel", "Region"), sum("Fresh"), sum("Milk"),
             sum("Grocery"), sum("Frozen"), sum("Detergents_Paper"), sum("Delicassen"))


######## Data Preparation ########

# Define the target variable
target <- "Channel"

# Definte the features
features <- setdiff(h2o.colnames(data), c("Channel"))

# Partition the data into training and test set (70/30)
samples <- h2o.splitFrame(data, c(0.7), seed=1) 
train_set <- samples[[1]]                    
test_set  <- samples[[2]]


######## Predictive Model ########

# Create GLM
# I chose Binomial because the predicted variable is binomial
glm_model1 <- h2o.glm(x = features, y = target, training_frame = train_set,
                      model_id = "glm_model1", family = "binomial")

# model summary	
print(summary(glm_model1))

# Evaluate on test data
perf_obj <- h2o.performance(glm_model1, newdata = test_set) 
h2o.accuracy(perf_obj, 0.95)
h2o.accuracy(perf_obj)

# Actual predictions
pred_creditability <- h2o.predict(glm_model1,test_set)
pred_creditability
