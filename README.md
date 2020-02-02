# Summary

This repository demonstrates the method for improving the stability of the Machine learning or Deep Learning model using a modified statistical significance metric.

Applications:

* User behavior analysis
* Noisy data
* Small training dataset

# Theory of the operation

## Algorithm basics

The major objective of cross-validation of the model is to estimate the model performance and provide tunning hyperparameters or feature engineering based on this.
For cases of a variety of feature behavior across users (using user-wise GroupKFold), noisy and/or small dataset with the weak feature-to-target signal, it's complicated to get the stable model configuration: the mean statistic across is well, but the variance of statistics is very high and varies depending on the validation set.

## Algorithm description

The theory behind Statscore is based on the classic measurement of statistical significance with the flexible coefficient for generalization ability.

Let's assume there are 10 users, the objective to predict will user pay the mortgage next month with. 
Training the model using GroupKFold across users returns 10 AUC scores, each score corresponds to each user in a list:

| User             | ROC-AUC             |
| --------------------- | ------------------- |
| 1     | 0.5    |
| 2    | 0.6  |
| 3  | 0.7  |
| ...           | ...               | 
| 10          | 0.8               | 

Instead of estimation of **mean AUC** across users which trends to overfitting to a particular user and not guarantees generalization on the overall population, an alternative approach is proposed:

![Statscore](/pictures/Statscore.PNG)

According to the formula, modeling is trending to estimate not only a mean of the score but a standard deviation as well: as higher mean, as higher the score; as higher std, as lower a score.

K - some additional generalization penalty (as higher K and lower influence of STD(AUC)).

The algorithm is utilizing bootstrap for a more accurate estimation of standard deviation and mean of the score distribution.

The objective of the model is to minimize some metric (e.g. RMSE), the parameter 'mode=' should be set as 'min':


```
score = Statscore(mode='min')

```
In this case, the input vector of the metric will be reversed to keep the equation above.

RMSE = 1/RMSE

Benefits of using this criterion for model performance estimation:
1. provide a stable model and prevents overfitting
2. provide an ability to cauterize users and train a separate model for them.

# Code examples

```

from StatScore import *

score = Statscore(mode='max')

auc_cv_vector = np.array([0.6,0.7,0.8,0.5,1]) #the vector of metrics per each validation trial, shape ([n,]), n- number of train/validation splits

score_estimated = score.evaluate(auc_vector)

```

