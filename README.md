# ai_coding_test solution

Project requirements :-

python 3.8.10
tensorflow 2.3.3



Steps to run project :

1. Put test images into test folder
2. run python3 main.py command 
3. above command will print file path and it's label whether it's apple to Not apple


Project Structure :-
1. training.py file contains model training related code
2. image.ipynb contains notebook file used in model development
3. data folder contains train & test Images data folders
4. output folder contains generated ML model after training


Training : -

run python3 training.py



Model Performance :-

Confusion Matrix->
 [[212   0]
 [0  159 ]]


Classification_report ->

  |    Name    |precision  |   recall |  f1-score  |  support |
  | :---       |     :---: |     ---: | ---:       |    ---:  |
  | apple      |    1.00   |    1.00  |    1.00    |   212    |
  | non-apple  |    1.00   |    1.00  |  1.00      |   159    |
  | accuracy   |           |          |  1.00      |   371    |
  | macro avg  |   1.00    |    1.00  |  1.00      |   371    |
  |weighted avg|   1.00    |    1.00  |  1.00      |   371    |











