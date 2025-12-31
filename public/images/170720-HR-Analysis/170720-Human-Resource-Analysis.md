
# HR Project

kaggle url : https://www.kaggle.com/ludobenistant/hr-analytics

프로젝트 목표 : 실제 기업의 인사 데이터를 분석하여, 퇴직할지 여부 맞추기

---
## Setting


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
```


```python
from sklearn.model_selection import KFold
from sklearn.cross_validation import train_test_split
```

---
## 데이터 전처리

---

** * 원본 Kaggle Data 불러와서 human으로 저장**


```python
human = pd.read_csv("HR_comma_sep.csv")
human.head(3)
```

** * Category로 나와있던 'sales', 'salary' column을 더미화**


```python
human['dep'] = human['sales']
human_dep = pd.get_dummies(human['sales'])
human_sal = pd.get_dummies(human['salary'])
del human['sales']
del human['salary']
```

** * multicollinearity을 위해 같은 카테고리에 있던 column 중 하나씩 삭제하여 human_mul로 저장**


```python
human = pd.concat([human,human_sal,human_dep],axis=1)
#multicollinearity을 위해 삭제해서 human_mul로 저장
human_mul = human.copy()
del human_mul['technical']
del human_mul['low']
```

** * 머신러닝을 돌릴 X feature들을 선정**


```python
feature_human =['satisfaction_level', 'last_evaluation', 'number_project',
       'average_montly_hours', 'time_spend_company', 'Work_accident',
       'promotion_last_5years', 'high', 'low', 'medium', 'IT', 'RandD',
       'accounting', 'hr', 'management', 'marketing', 'product_mng', 'sales',
       'support', 'technical']
```

** * 모델 검증을 위하여 Train Set / Test Set 나누기**


```python
myTrain, myTest = train_test_split(human,test_size = 0.1)
X_Train = myTrain[feature_human]
Y_Train = myTrain['left']

X_Test = myTest[feature_human]
Y_Test = myTest['left']
```

---
## Model 1. 로지스틱 분석

---
** * Scikit Learn을 활용한 로지스틱 분석**


```python
from sklearn.linear_model import LogisticRegression
```


```python
logreg = LogisticRegression(C=1e5)
logreg.fit(X_Train, Y_Train)
```




    LogisticRegression(C=100000.0, class_weight=None, dual=False,
              fit_intercept=True, intercept_scaling=1, max_iter=100,
              multi_class='ovr', n_jobs=1, penalty='l2', random_state=None,
              solver='liblinear', tol=0.0001, verbose=0, warm_start=False)



미리 빼놓은 10%의 Test Set에 대한 정확도


```python
logreg.score(X_Test,Y_Test)
```




    0.79400000000000004



Cross Validation


```python
Logit_score = cross_val_score(logreg,
                              X_Test.as_matrix(),
                              Y_Test.as_matrix(),
                              cv=5)
Logit_score.mean()
```




    0.78799591847316819



---
** * Statsmodel을 활용한 로지스틱 분석**


```python
# logistic regression
logit = sm.Logit(Y_Train, X_Train) # Survived는 목적 변수

# fit the model
result = logit.fit() 

# result.params # 분석결과 출력
# odds rate
print (np.exp(result.params))
```

    Optimization terminated successfully.
             Current function value: 0.429357
             Iterations 7
    satisfaction_level       0.015943
    last_evaluation          2.112190
    number_project           0.728021
    average_montly_hours     1.004419
    time_spend_company       1.317232
    Work_accident            0.213670
    promotion_last_5years    0.251191
    high                     0.242803
    low                      1.553492
    medium                   0.912031
    IT                       0.820945
    RandD                    0.530398
    accounting               0.984193
    hr                       1.234437
    management               0.635344
    marketing                1.001736
    product_mng              0.879852
    sales                    0.993661
    support                  1.083700
    technical                1.078422
    dtype: float64


---
## Model 2. Decision Tree
---

** * Scikit learn을 활용한 의사결정나무 만들기**


```python
from sklearn import tree
```


```python
DT_model = tree.DecisionTreeClassifier(max_depth = 10, min_samples_leaf = 15)
DT_model.fit(X_Train, Y_Train)
DT = pd.DataFrame(DT_model.predict_proba(X_Test))
DT['left'] = Y_Test.reset_index()['left']
DT.head()
tree.DecisionTreeClassifier()
```




    DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
                max_features=None, max_leaf_nodes=None,
                min_impurity_split=1e-07, min_samples_leaf=1,
                min_samples_split=2, min_weight_fraction_leaf=0.0,
                presort=False, random_state=None, splitter='best')



** * 모델 검증**

전체 데이터셋에 대한 정확도


```python
correct_num = 0
for item in DT.values:
    if item[0]>item[1]:
        temp = 0
    else: temp = 1
    
    if item[2]==temp: correct_num+=1
print(correct_num/len(myTest))
```

    0.97


Cross Validation Score 확인


```python
DT_score = cross_val_score(DT_model,X_Test.as_matrix(),Y_Test.as_matrix(),cv=5)
DT_score.mean()
```




    0.95535071130419946



---
## Model 3. 랜덤포레스트
---

** * Scikit learn에서 RandomForest Classifier 불러오기**


```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
pd.set_option("display.max_columns",200)
```

분석에 사용할 X feature들 선정


```python
feature_human =['satisfaction_level', 'last_evaluation', 'number_project',
       'average_montly_hours', 'time_spend_company', 'Work_accident',
       'promotion_last_5years', 'high', 'low', 'medium', 'IT', 'RandD',
       'accounting', 'hr', 'management', 'marketing', 'product_mng', 'sales',
       'support', 'technical']
```

** * 하이퍼파라미터 최적화**
- 반복문을 이용해 최적의 모델 찾기
- while문을 돌리며 지정된 횟수만큼 모델의 성능을 update


```python
RFTest_size = 0.25
n = 4
print("RandomForestClasifier Start.")
x_train, x_validate, y_train, y_validate = train_test_split(X_Train,Y_Train,test_size = RFTest_size, 
                                                    random_state=39,stratify= Y_Train)

i = 0
#my_model = RandomForestClassifier(n_estimators = 1000, criterion = "gini",min_samples_split = 5, min_samples_leaf = 5, warm_start = True)

RF_model = RandomForestClassifier(n_estimators = 1000,n_jobs=-1,max_features=7)
RF_model.fit(x_train, y_train)
RF_probs = RF_model.predict_proba(x_validate)
RF_logloss = log_loss(y_validate, RF_probs)
print(str(i) + ": "+ str(RF_logloss))

while(True):
    #temp_model_cat = RandomForestClassifier(n_estimators = 1000, criterion = "gini",min_samples_split = 5, min_samples_leaf = 5, warm_start = True)
    temp_model = RandomForestClassifier(n_estimators = 1000,n_jobs=-1,max_features=7)
    temp_model.fit(x_train, y_train)
    temp_probs = temp_model.predict_proba(x_validate)
    
    if RF_logloss > log_loss(y_validate, temp_probs):
        i += 1
        RF_model = temp_model
        RF_logloss = log_loss(y_validate, temp_probs)
        print(str(i) + ": "+ str(RF_logloss))
    # 종료조건    
    if i >= n:
        break    
        
print("Complete.")
```

    RandomForestClasifier Start.
    0: 0.0556763920555
    1: 0.0556297446114
    2: 0.0547819268539
    3: 0.0547341809308
    4: 0.0546790442451
    Complete.


** * 결과 분석 : 중요한 변수가 뭔지 알아보자**


```python
fi = RF_model.feature_importances_
# make importances relative to max importance
fi = 100.0 * (fi / fi.max())
sorted_idx = np.argsort(fi)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.figure(figsize=(10, 15))
plt.subplot(1, 1, 1)
plt.barh(pos, fi[sorted_idx], align='center')
plt.yticks(pos, x_train.columns[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()
```


![png](output_43_0.png)


** * 모델 검증**


```python
RF_result = RF_model.predict_proba(X_Test)
RF = pd.DataFrame(RF_result)
RF['left'] = Y_Test.reset_index()['left']
```

전체 Data에 대한 정확도


```python
RF_model.score(X_Test,Y_Test)
```




    0.98933333333333329



Cross validation 정확도


```python
RF_score = cross_val_score(RF_model, X_Test.as_matrix(),Y_Test.as_matrix(), cv=10)
print("RandomForest Results: %.2f%% (%.2f%%)" % (RF_score.mean()*100, RF_score.std()*100))
```

    RandomForest Results: 97.07% (1.60%)


---
## Model 3. Keras_binomial Classification

Neural Network의 python framework인 keras를 이용하여 binomial classification을 시도해보겠습니다.

튜토리얼 단계로, 그냥 따라했을 뿐입니다.

참고 : http://machinelearningmastery.com/binary-classification-tutorial-with-the-keras-deep-learning-library/

---

** * Keras를 활용한 Binomial Classification**


```python
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
```

    Using TensorFlow backend.



```python
from sklearn.metrics import accuracy_score
```


```python
encoder = LabelEncoder()
_X = X_Train.values.astype(float)
_Y = Y_Train.values
encoder.fit(_Y)
encoded_Y = encoder.transform(_Y)
```


```python
def create_baseline():
    # create model
    model = Sequential()
    model.add(Dense(20, input_dim=20, init='normal', activation='relu'))
    model.add(Dense(1, init='normal', activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
```

** * 모델 검증**


```python
estimator = KerasClassifier(build_fn=create_baseline, nb_epoch=3, batch_size=5, verbose=0)
results = cross_val_score(estimator, _X, encoded_Y, cv=3)
print("KerasClassifier Results: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
```

    /Users/sungnampark/anaconda/lib/python3.6/site-packages/ipykernel_launcher.py:4: UserWarning: Update your `Dense` call to the Keras 2 API: `Dense(20, input_dim=20, activation="relu", kernel_initializer="normal")`
      after removing the cwd from sys.path.
    /Users/sungnampark/anaconda/lib/python3.6/site-packages/ipykernel_launcher.py:5: UserWarning: Update your `Dense` call to the Keras 2 API: `Dense(1, activation="sigmoid", kernel_initializer="normal")`
      """


    KerasClassifier Results: 86.92% (1.50%)


---
