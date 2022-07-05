PSET2 - House Price Prediction
==============================

MLOps project for PSET2-House Price Prediction

This project is to predict the price of a house given various factors like no. of bedrooms, total no. of rooms, median income of that locality,
population, number of households, age of the house. The target here is to predict the median price of houses. All the variables here are numeric and
Lasso Regression is used to get the predictions.

Aim of this project is not to optimize the ML model, rather show end-to-end MLOps process-starting from ingesting the raw data, splitting
into train-test set, model building and model version logging. Below are the steps mentioned:

Step 1:
Create a github repo for code version controlling. Here is this project's github repo - https://github.com/AnweshaMohanty-07/MLOps-Project/tree/main/PSET2

Step 2:
Create an account in AWS with EC2 instance. For this project Cloud9 is used to create the project folder. 
Here is the AWS instance link - https://us-west-2.console.aws.amazon.com/cloud9/ide/23706761fd5540b9a2ce1d48d744a08e
Clone the github repo in the environment so that all changes made in AWS can be pushed to github.- git clone githubrepo

Step 3:
Use cookie cutter to create standard Data Science project folders.
pip install cookiecutter
cookiecutter https://github.com/drivendata/cookiecutter-data-science

Step 4:
Create the makefile and requirements.txt to install all necessary libraries and their dependencies.
Upload the raw file to data/raw

Step 5:
To maintain data versioning, DVC is used here. It will create a metadata file of the original data which will be maintained by github 
instead of storing the complete data.
pip install dvc
dvc init 

Step 6:
Dagshub has been used to maintain one repository that will easily facilitate the MLOps pipeline. It has instances of DVC, MLFlow and Github.
All the changes pushed to github will be visible to in Dagshub as well. One can create an account in Dagshub and the replicate the current
github repo.

Step 7:
For model registry and model versioning, MLFlow is used which is again linked on Dagshub. Each time an iteration is run,
models along with the parameters are logged which can be easily reproducible.

Step 8:
All python codes are maintained in src folder. A params.yaml is maintained that keeps track of all the arguments passed to functions.
Data and its versions are maintained in data folder.
artifacts folder has the versions of models executed.

The above steps outline the CI pipeline using AWS,Github, DVC, MLFlow and Dagshub.
