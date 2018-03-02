---
layout: post
title:  "Allstate Insurance Claim Value"
date:   2018-3-2 14:33:00 -0500
categories: datascience
excerpt: The following is my solution to a Kaggle contest that ran in early 2017 regarding predicting the value of insurance claim payout as a function of many different given features.  For privacy purposes (and also to protect internal information I assume), all of the variables are anonymized.  There are a mix of both continuous and categorical variables, with the latter providing the vast majority of the features in the dataset.  In order to treat the categorical variables, we must 'one-hot-encode' them, so for each combination of categorical variable and possible value for that variable, we create a new column with either a 0 if the entry doesn't have this characteristic or 1 if it does.
---
#### The following is my solution to a Kaggle contest that ran in early 2017 regarding predicting the value of insurance claim payout as a function of many different given features.  For privacy purposes (and also to protect internal information I assume), all of the variables are anonymized.  There are a mix of both continuous and categorical variables, with the latter providing the vast majority of the features in the dataset.  In order to treat the categorical variables, we must 'one-hot-encode' them, so for each combination of categorical variable and possible value for that variable, we create a new column with either a 0 if the entry doesn't have this characteristic or 1 if it does.


```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

plt.rcParams["patch.force_edgecolor"] = True
%matplotlib inline
```


```python
train = pd.read_csv("train.csv")
train.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>cat1</th>
      <th>cat2</th>
      <th>cat3</th>
      <th>cat4</th>
      <th>cat5</th>
      <th>cat6</th>
      <th>cat7</th>
      <th>cat8</th>
      <th>cat9</th>
      <th>...</th>
      <th>cont6</th>
      <th>cont7</th>
      <th>cont8</th>
      <th>cont9</th>
      <th>cont10</th>
      <th>cont11</th>
      <th>cont12</th>
      <th>cont13</th>
      <th>cont14</th>
      <th>loss</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>A</td>
      <td>B</td>
      <td>A</td>
      <td>B</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>B</td>
      <td>...</td>
      <td>0.718367</td>
      <td>0.335060</td>
      <td>0.30260</td>
      <td>0.67135</td>
      <td>0.83510</td>
      <td>0.569745</td>
      <td>0.594646</td>
      <td>0.822493</td>
      <td>0.714843</td>
      <td>2213.18</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>A</td>
      <td>B</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>B</td>
      <td>...</td>
      <td>0.438917</td>
      <td>0.436585</td>
      <td>0.60087</td>
      <td>0.35127</td>
      <td>0.43919</td>
      <td>0.338312</td>
      <td>0.366307</td>
      <td>0.611431</td>
      <td>0.304496</td>
      <td>1283.60</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>A</td>
      <td>B</td>
      <td>A</td>
      <td>A</td>
      <td>B</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>B</td>
      <td>...</td>
      <td>0.289648</td>
      <td>0.315545</td>
      <td>0.27320</td>
      <td>0.26076</td>
      <td>0.32446</td>
      <td>0.381398</td>
      <td>0.373424</td>
      <td>0.195709</td>
      <td>0.774425</td>
      <td>3005.09</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10</td>
      <td>B</td>
      <td>B</td>
      <td>A</td>
      <td>B</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>B</td>
      <td>...</td>
      <td>0.440945</td>
      <td>0.391128</td>
      <td>0.31796</td>
      <td>0.32128</td>
      <td>0.44467</td>
      <td>0.327915</td>
      <td>0.321570</td>
      <td>0.605077</td>
      <td>0.602642</td>
      <td>939.85</td>
    </tr>
    <tr>
      <th>4</th>
      <td>11</td>
      <td>A</td>
      <td>B</td>
      <td>A</td>
      <td>B</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>A</td>
      <td>B</td>
      <td>...</td>
      <td>0.178193</td>
      <td>0.247408</td>
      <td>0.24564</td>
      <td>0.22089</td>
      <td>0.21230</td>
      <td>0.204687</td>
      <td>0.202213</td>
      <td>0.246011</td>
      <td>0.432606</td>
      <td>2763.85</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 132 columns</p>
</div>



#### Let's first check if there's any missing values in the dataset:


```python
train.isnull().sum().sum()
```




    0



#### Let's now take a look at the distributions for the continuous variables.  Note that I've added a plot that transforms the claim loss to log(loss+1).  The reason for this is that the log function fixes the skewness of the loss distribution, and this will make the regression perform better.  I choose log(loss+1) instead of just log(loss) to cover the case if loss = 0 and to assure the transformed variable is positive (recall that the log of a number between 0 and 1 is negative).  If some of the target variable space is negative, this will also throw off the regression.  It's possible that some of the other continuous features would benefit from this treatment as well, but I'll leave that to another time.  Of course, once the prediction is made, the inverse function exp(x)-1 will transform the variable back to dollars (presumably this is the unit, we are never told).


```python
plt.subplots(4,4,figsize=(12,12))
for i in range(1,15):
    plt.subplot(4,4,i)
    plt.hist(train["cont%i"%i],bins=20)
    plt.xlabel("cont%i"%i)
    
plt.subplot(4,4,15)
plt.hist(train["loss"],bins=200)
plt.xlim(0,15000)
plt.xlabel("loss")

plt.subplot(4,4,16)
plt.hist(np.log1p(train["loss"]),bins=200)
# plt.xlim(0,15000)
plt.xlabel("log(loss+1)")
```




    <matplotlib.text.Text at 0x114570ad0>




![png](/images/allstate/output_6_1.png)


#### Now let's check out the correlation matrix for all the continuous variables:


```python
contVars = ["cont"+"%i"%i for i in range(1,15)]
contVars.append("loss")
plt.figure(figsize=(14,14))
sns.heatmap(train[contVars].corr(),annot=True,cmap="jet")
```




    <matplotlib.axes._subplots.AxesSubplot at 0x111f03750>




![png](/images/allstate/output_8_1.png)


#### Now, let's do the one-hot-encoding of the training and test samples.  I temporarily merge the training and test samples in order to do this because what can (and does in this case) happen is that the test sample will have options for some categorical variables that the training does not have, so the columns in each case would be different.


```python
train["isTest"]=False

test = pd.read_csv("test.csv")
test["isTest"] = True
test["loss"] = np.pi
a = pd.concat([train,test])
a = pd.get_dummies(a,drop_first=True)
train = a[a.isTest==False].copy()
test = a[a.isTest==True].copy()
train.drop("isTest",axis=1,inplace=True)
test.drop("isTest",axis=1,inplace=True)
del a
test.drop("loss",axis=1,inplace=True)
```

#### Split up the training sample into a training and test sample in order to train and test the regression.  Remember that the test sample we defined in the cell above was the one given by Kaggle *without the target variable*, so we can't use it to test our model.  Also recall that I transform the loss to log(loss + 1) in order to correct the skewness of the distribution.


```python
from sklearn.model_selection import train_test_split

X = train.drop("loss",axis=1).drop("id",axis=1)
y = np.log1p(train["loss"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

#### I use a Linear Support Vector Regressor (LinearSVR) to model the dependence of the loss upon the features given in the dataset.  Remember that, of course, after making a prediction with the regression, we must invert the transformation with exp(y) - 1.  The MAE I get is competitive in the Kaggle competition, which is nice for such a quick exercise!


```python
from sklearn.svm import LinearSVR
from sklearn.metrics import *

clf = LinearSVR()
clf.fit(X_train,y_train)
pred = clf.predict(X_test)

y_testTrans = np.expm1(y_test)
predTrans = np.expm1(pred)

print mean_absolute_error(y_testTrans,predTrans)
print mean_squared_error(y_testTrans,predTrans)
print np.sqrt(mean_squared_error(y_testTrans,predTrans))

testX = test.drop("id",axis=1)
testPred = np.expm1(clf.predict(testX))
test["loss"] = testPred
test[["id","loss"]].to_csv("linearSVR.csv",index=False)
test.drop("loss",axis=1,inplace=True)
```

    1269.39779994
    7593361.65182
    2755.6054964


#### For my last step, I'll perform cross-validation of the model across the training sample to see how well the model generalizes.  Since the scores are all similar in value, we're doing well on this front.


```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(clf, X_train, y_train, cv=5)
print scores
```

    [ 0.50818537  0.50933825  0.52031112  0.51211244  0.51128369]



```python

```
