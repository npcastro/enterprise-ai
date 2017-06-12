#Installation
#Download the training dataset from https://www.kaggle.com/c/dogs-vs-cats/data.
#Download the EBImage package to be used for image preprocessing

source("http://bioconductor.org/biocLite.R")
biocLite()
biocLite("EBImage")
#Install package pbapply. pbapply is same as lapply function but adds a progress bar which helps keep a track of the activity

install.packages("pbapply")
#EBImage provides general purpose functionality for image processing and analysis.
#You can read more about the package at https://www.r-bloggers.com/r-image-analysis-using-ebimage/


library(EBImage)
library(pbapply)
library(caret)
library(mxnet)

#Set the image directory
image_dir_cat <- "C:\\Convolutional_Neural_Networks\\dataset\\cats"
image_dir_dog <- "C:\\Convolutional_Neural_Networks\\dataset\\dogs"

#Load a image and verify if the image is loaded properly
example_cat_image <- readImage(file.path(image_dir_cat, "cat.1.jpg"))
example_dog_image <- readImage(file.path(image_dir_dog, "dog.1.jpg"))
display(example_cat_image)
display(example_dog_image)

# As we are going to use images just to classify cats and dogs, we can transform the image
# to grey scale. This will help to load the images to R directly. We will use EBImage to resize the images to 28×28
# Each image will be turned into a vector of length 784, with each element representing the value in a pixel.

width <- 28
height <- 28

#Function to extract the image features and store them as a feature matrix.

extract_feature <- function(dir_path, width, height, is_cat = TRUE, add_label = TRUE) {
  img_size <- width*height
  ## List images in path
  images_names <- list.files(dir_path)
  if (add_label) {
    ## Select only cats or dogs images
    images_names <- images_names[grepl(ifelse(is_cat, "cat", "dog"), images_names)]
    ## Set label, cat = 0, dog = 1
    label <- ifelse(is_cat, 0, 1)
  }
  print(paste("Start processing", length(images_names), "images"))
  ## This function will resize an image, turn it into greyscale
  feature_list <- pblapply(images_names, function(imgname) {
    ## Read image
    img <- readImage(file.path(dir_path, imgname))
    ## Resize image
    img_resized <- resize(img, w = width, h = height)
    ## Set to grayscale
    grayimg <- channel(img_resized, "gray")
    ## Get the image as a matrix
    img_matrix <- grayimg@.Data
    ## Coerce to a vector
    img_vector <- as.vector(t(img_matrix))
    return(img_vector)
  })
  ## bind the list of vector into matrix
  feature_matrix <- do.call(rbind, feature_list)
  feature_matrix <- as.data.frame(feature_matrix)
  ## Set names
  names(feature_matrix) <- paste0("pixel", c(1:img_size))
  if (add_label) {
    ## Add label
    feature_matrix <- cbind(label = label, feature_matrix)
  }
  return(feature_matrix)
}

#Save the features of images into data.frame

cats_data <- extract_feature(dir_path = image_dir_cat, width = width, height = height)
dogs_data <- extract_feature(dir_path = image_dir_dog, width = width, height = height, is_cat = FALSE)
#Check the dimensions
dim(cats_data)
dim(dogs_data)

#Model Training
#Data partitions: randomly split 90% of data into training set with equal weights for cats and dogs, 
#and the rest 10% will be used as the test set.

## Bind rows in a single dataset
complete_set <- rbind(cats_data, dogs_data)

## test/training partitions
training_index <- createDataPartition(complete_set$label, p = .9, times = 1)
training_index <- unlist(training_index)
train_set <- complete_set[training_index,]
dim(train_set)
test_set <- complete_set[-training_index,]
dim(test_set)

# Fix train and test datasets
train_data <- data.matrix(train_set)
train_x <- t(train_data[, -1])
train_y <- train_data[,1]
train_array <- train_x
dim(train_array) <- c(28, 28, 1, ncol(train_x))

test_data <- data.matrix(test_set)
test_x <- t(test_set[,-1])
test_y <- test_set[,1]
test_array <- test_x
dim(test_array) <- c(28, 28, 1, ncol(test_x))

#Model:

mx_data <- mx.symbol.Variable('data')
# 1st convolutional layer has 20 filters. The size of each filter is n*n(5*5 in our case) and is
# genearlly less than the size of the image
conv_1 <- mx.symbol.Convolution(data = mx_data, kernel = c(5, 5), num_filter = 20)
tanh_1 <- mx.symbol.Activation(data = conv_1, act_type = "tanh")
#  Next define a Pooling layer. Also known as downsample layer. This basically takes a filter (normally of size 2x2)
#  and a stride of the same length. It then applies it to the input volume and outputs the maximum number in every subregion 
#  that the filter convolves around. We have used max pooling. Other type of pooling includes avg for average and sum for sum pooling
pool_1 <- mx.symbol.Pooling(data = tanh_1, pool_type = "max", kernel = c(2, 2), stride = c(2,2 ))
#Similarily define a second convolution and pooling layer
# 2nd convolutional layer 5x5 kernel and 50 filters.
conv_2 <- mx.symbol.Convolution(data = pool_1, kernel = c(5,5), num_filter = 50)
tanh_2 <- mx.symbol.Activation(data = conv_2, act_type = "tanh")
pool_2 <- mx.symbol.Pooling(data = tanh_2, pool_type = "max", kernel = c(2, 2), stride = c(2, 2))
# 1st fully connected layer
# Flatten the structure so that it can be an input to the Fully connected network layer
flat <- mx.symbol.Flatten(data = pool_2)
fcl_1 <- mx.symbol.FullyConnected(data = flat, num_hidden = 500)
tanh_3 <- mx.symbol.Activation(data = fcl_1, act_type = "tanh")
# 2nd fully connected layer
fcl_2 <- mx.symbol.FullyConnected(data = tanh_3, num_hidden = 2)
# Output
NN_model <- mx.symbol.SoftmaxOutput(data = fcl_2)

# Set seed for reproducibility
mx.set.seed(100)

device <- mx.cpu()

# Train the model
model <- mx.model.FeedForward.create(NN_model, X = train_array, y = train_y,
                                     ctx = device,
                                     num.round = 30,
                                     array.batch.size = 100,
                                     learning.rate = 0.05,
                                     momentum = 0.9,
                                     wd = 0.00001,
                                     eval.metric = mx.metric.accuracy,
                                     epoch.end.callback = mx.callback.log.train.metric(100))

# Test set
predict_probs <- predict(model, test_array)
predicted_labels <- max.col(t(predict_probs)) - 1
table(test_data[, 1], predicted_labels)