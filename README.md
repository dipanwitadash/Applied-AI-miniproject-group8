# Applied-AI-miniproject-group8

Anusha Talapaneni (anutal-2@student.ltu.se),
Dipanwita Dash (dipdas-2@student.ltu.se),
Meenakshi Subhash Chippa (meechi-2@student.ltu.se)

## Project : Multivariate Time Series Forecasting with LSTMs  (Air Pollution Forecasting)

YouTube link: https://www.youtube.com/watch?v=y-r6IXs00oo

The presentation document in pdf format, is part of GitHub as well - D7041E_Miniproject_Group8.pdf

## Instruction for executing the notebooks
All notebooks were created and execuetd in kaggle. If someone wants to reexecute, we recommend to do directly in kaggle with the steps as follows:
* Download the notebooks from github
* Import the notebooks in Kaggle (You need to have a kaggle account or register yourself)
* Download the dataset and import the same in kaggle
* Make sure to keep the directory name same "air-pollution-data" as in the code for the dataset 
* The code should run as it is after that
In order to execute in local PC, the dependencies need to be installed through PIP.

## Project Introduction
Aim of our project is to “forecast air pollution” using LSTM which is type of RNN.

Air Quality dataset is used which contains features like  pollution, dew point, temperature, pressure, wind direction, speed, date-time , snow and rain.

Our project covers various experiments which involves various steps in general:

* To transform a raw dataset into something that we can use for time series forecasting.

* To prepare data and fit an LSTM for a multivariate time series forecasting problem.

* To make a forecast on test data and rescale the result back into the original units.

## Dataset - pollution.csv

## 1st Experiment (d7041e-project-grp8-exp1.ipynb)
Data Preprocessing:

* Parsing Dates (datetime column)
* Dropping Unnecessary Column
* Renaming Columns: e.g., 'pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain
* Handling Missing Values
* Discarding Initial Rows: The first 24 hours of data are dropped, as they may not be relevant for analysis.
* Label Encoding: The 'wnd_dir' (wind direction) column, which is categorical, is encoded into numeric format using LabelEncoder.
* Data Normalization: The features are normalized using MinMaxScaler to scale them to a range between 0 and 1.

	Transforming Data for Supervised Learning:
The supervised learning problem will be framed as predicting pollution at the current hour (t) based on previous pollution measurements and weather conditions..

Reshaping for LSTM Model: 
The input data is reshaped into a 3D format required by the LSTM model, which is [samples, timesteps, features].

Model Building and Training:
LSTM model with 50 neurons &  Adam SGD optimiser & Epochs = 50

Making Predictions and Inverting Transformations: 
The model is used to make predictions on the test set. These predictions, along with the test set, are then transformed back to their original scale.

Result : 
RMSE which is the difference between the predicted values and the actual values in the test dataset.
RMSE of 26.496
https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/

## 2nd  Experiment (d7041e-project-grp8-exp2.ipynb)
Model Building and Training: 
LSTM model with 50 neurons &  Adam SGD optimiser & Epochs = 50
Early stopping
Batch_Size : 72
Data splitted : Train,val,test

## 3rd Experiment (d7041e-project-grp8-exp3.ipynb)
Hyperparameter Tuning:
Experimented  with different numbers of LSTM units and optimizers also used different random seeds for reproducibility to ensure the robustness of the model performance
 
lstm_units = [30, 50, 100] # Neurons
optimizers = ['adam', 'rmsprop']
seeds = [42, 7, 21]

Enhanced Model Evaluation:
 In addition to calculating the RMSE, the code also computes the Mean Absolute Error (MAE) and the R² score, providing a more comprehensive evaluation of model performance.

## 4th Experiment (d7041e-project-grp8-exp4.ipynb)
Model Parameters:
Data splitted : Train,Validation,Test
Random Forest:
n_estimators=100
loss function: Mean Squared Error (MSE)

Support Vector Regressor:
	kernel  = rbf
	degree=3 (polynomial kernel function)
	C=1.0
	epsilon=0.1

## Results

Experiment 1: Test RMSE of 26.496

Experiment 2: Test RMSE: 26.185

Experiment 3: Test RMSE: 26.371
                  Mean Absolute Error: 14.037 
                  R² Score: 0.918
                       
Experiment 4: SVM RMSE : 83.06
        Random Forest RMSE : 74.71

#### Please note - Experiment 2 was rerun after recording the presentation and the RMSE changed to 26.403 since there was no seed defined. 
## Conclusion
* Thorough exploration of multivariate time series forecasting

* Detailed analysis of LSTM networks and comparison with traditional models

* Results highlighting LSTM's capability with a strong R² score of 0.918

* RMSE values indicating the challenging nature of environmental forecasting

* Insightful comparative analysis against Random Forest and Support Vector Regressor models

* Potential practical applications in environmental monitoring and policy making.

