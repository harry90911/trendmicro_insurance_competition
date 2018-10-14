## 匯入套件
# pandas & numpy
import pandas as pd
import numpy as np

#調整表格預覽結果
pd.set_option("display.max_columns",999)
pd.set_option("display.max_rows",10)

# logistic model 所用到的 package
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import model_selection
from sklearn.model_selection import cross_val_score

# 決策樹 所用到的 package
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
from sklearn.tree import export_graphviz
from IPython.display import Image  
import pydotplus

# 計算平均數
from statistics import mean 
import statsmodels.api as sm


## 匯入資料
# 1. 更改路徑，直接讀入。
# 2. 更改路徑之後，讀一次就好，不用一直重新讀資料。

# policy = pd.read_csv("/Volumes/Transcend 1/客戶續約金額預測/policy_claim/policy_0702.csv", header=0)
# claim = pd.read_csv("/Volumes/Transcend 1/客戶續約金額預測/policy_claim/claim_0702.csv", header=0)
# train = pd.read_csv("/Volumes/Transcend 1/客戶續約金額預測/training-set.csv", header=0)
# test = pd.read_csv("/Volumes/Transcend 1/客戶續約金額預測/testing-set.csv", header=0)

# policy 和 training 合併
policy_train = pd.merge(policy, train, how='left', on='Policy_Number')

#claim 和 policy 和 training 合併，創造變數：claim，如果該案子有出保，則為 1，否則為 0
policy_claim = pd.merge(policy, claim, how="left", left_on=['Policy_Number','Insurance_Coverage'], right_on=['Policy_Number','Coverage'])

policy_claim_train = pd.merge(policy_claim, train, on="Policy_Number", how='inner')
policy_claim_train["claim"] = pd.isnull(policy_claim_train["Claim_Number"])
policy_claim_train["claim"] = policy_claim_train["claim"].astype(int).replace({True:0, False:1})

# policy 和 test 合併，用來預測最後結果
policy_claim_test = pd.merge(policy_claim, test, on="Policy_Number", how="inner")
policy_claim_test["claim"] = pd.isnull(policy_claim_test["Claim_Number"])
policy_claim_test["claim"] = policy_claim_test["claim"].astype(int).replace({True:0, False:1})



# 將資料整理寫成函式
# 寫成函式的目的在於「用於訓練的資料」和「用來預測的資料」要「分別」進行進行資料整理，才不會所有資料混在一起。
# 「用於訓練的資料」： policy_training
# 「用來預測的資料」： policy_test

## 進行 policy_training 的資料整理的 function
def data_preprocessing_policy(input_dataset):
    dataset = input_dataset.copy()
    
    # 被解釋變數：Next_Premium(續約金額) / Renewal(是否續約)
    dataset["Next_Premium"] = pd.to_numeric(dataset["Next_Premium"])
    dataset["Renewal"] = np.where(dataset["Next_Premium"]>0, 1, 0)
    
    #是否為國產車
    index = dataset["Imported_or_Domestic_Car"].value_counts(normalize=True).reset_index()["index"].tolist()
    prob = dataset["Imported_or_Domestic_Car"].value_counts(normalize=True).reset_index()["Imported_or_Domestic_Car"].tolist()
    dataset["Imported_or_Domestic_Car"].fillna(np.random.choice(index, p=prob), inplace=True)
    dataset.loc[dataset["Imported_or_Domestic_Car"]!=10, "Imported_or_Domestic_Car_dummy"] = "非國產車"
    dataset.loc[dataset["Imported_or_Domestic_Car"]==10, "Imported_or_Domestic_Car_dummy"] = "國產車"
    dummies = pd.get_dummies(dataset["Imported_or_Domestic_Car_dummy"])
    dataset = dataset.join(dummies["國產車"])
    
    #是否曾經續約過
    dataset["Cancellation"].replace(" ", 1, inplace=True)
    dataset["Cancellation"].replace("Y", 0, inplace=True)
    dataset.rename(columns={"Cancellation":"Last_Renewal"}, inplace=True)

    #主要險種分類
    dummies = pd.get_dummies(dataset["Main_Insurance_Coverage_Group"])
    dataset = dataset.join(dummies)
    
    #被保險人身份
    dataset["fsex"].fillna("法人", inplace=True)
    dataset["fsex"].replace(r'\s+', "法人", regex=True, inplace=True)
    dataset["fsex"].replace("1", "男", inplace=True)
    dataset["fsex"].replace("2", "女", inplace=True)
    dummies = pd.get_dummies(dataset["fsex"])
    dataset = dataset.join(dummies)
    
    #被保險人是否結婚
    dataset["fmarriage"].replace(r'\s+', np.random.choice(["1", "2"], p=[2/3,1/3]), regex=True, inplace=True)
    dataset["fmarriage"].replace({"1":"已婚", "2":"未婚"}, inplace=True)
    dummies = pd.get_dummies(dataset["fmarriage"])
    dataset = dataset.join(dummies)
    
    #是否持有其他保單
    dataset["Dummy_TmNewa"] = np.where(dataset["Multiple_Products_with_TmNewa_(Yes_or_No?)"]>0, 1, 0)
    
    #車子出廠年數
    dataset["Manafactured_Year_and_Month"] = (2015 - dataset["Manafactured_Year_and_Month"])
    
    #年齡
    column = []
    for i in dataset["ibirth"].str.split("/").tolist():
        try:
            column.append(i[1])
        except TypeError:
            #有遺漏值
            column.append(i)
    dataset["iyear"] = column
    dataset["age"] = (2015 - pd.to_numeric(dataset["iyear"]))
    dataset["age"].isnull().value_counts()
    dataset["age"].fillna(dataset["age"].mode()[0], inplace=True)
    dataset["age"] = dataset["age"].astype("int")
    
    return dataset

## 進行 claim_policy_training 的資料整理的 function
def data_preprocessing_claim(input_dataset):
    dataset = input_dataset.copy()
    def Nature_of_the_claim(x):
        if x==1:
            return 1
        if x==2:
            return 0
    dataset['Nature_of_the_claim']=claim['Nature_of_the_claim'].apply(Nature_of_the_claim)

    def Gender_of_driver(x):
        if x==1:
            return 1
        if x==2:
            return 0
    dataset["Driver's_Gender"] = dataset["Driver's_Gender"].apply(Gender_of_driver)

    included_pairs = sorted(dataset["Driver's_Relationship_with_Insured"].unique ())
    pair_dums = pd.get_dummies(dataset["Driver's_Relationship_with_Insured"].astype('category'), prefix="Driver's_Relationship_with_Insured")
    dataset = pd.concat ([dataset , pair_dums] , axis=1)

    dataset["DOB_of_Driver"]=pd.to_datetime(dataset["DOB_of_Driver"])
    dataset["DOB_of_Driver"]=2018-dataset["DOB_of_Driver"].dt.year

    def Marital_Status_of_Driver(x):
        if x==1:
            return 1
        if x==2:
            return 0
    dataset["Marital_Status_of_Driver"]=dataset["Marital_Status_of_Driver"].apply(Marital_Status_of_Driver)

    def Cause_of_Loss(x):
        if x=="17ba0791499db908433b80f37c5fbc89b870084b":
            return 1
        else:
            return 0
    dataset["Cause_of_Loss_1"]=dataset["Cause_of_Loss"].apply(Cause_of_Loss)

    def Salvage_or_Subrogation(x):
        if x==0:
            return 1
        else:
            return 0
    dataset["Salvage_or_Subrogation_1"]=dataset["Salvage_or_Subrogation?"].apply(Salvage_or_Subrogation)
    
    return dataset

## 進行 policy_training 的資料整理的 function
def create_dataset_policy(input_dataset):
    dataset = input_dataset.copy()
    data_Y = dataset["Renewal"]
    #basic
    #dataset["車責"]
    #dataset["男"]
    #dataset["未婚"]
    #dataset["非國產車"]
    data_X = pd.DataFrame([
                dataset["Premium"],
                dataset["Last_Renewal"],
                dataset["竊盜"],
                dataset["車損"],
                dataset["女"],
                dataset["法人"],
                dataset["已婚"],
                dataset["Dummy_TmNewa"],
                dataset["lia_class"], 
                dataset["plia_acc"], 
                dataset["pdmg_acc"],
                dataset["claim"],
                dataset["Engine_Displacement_(Cubic_Centimeter)"],
                dataset["Manafactured_Year_and_Month"],
                dataset["age"],
                dataset["Replacement_cost_of_insured_vehicle"],
                dataset["國產車"],
                dataset["Coverage_Deductible_if_applied"],
                dataset["Insured_Amount1"],
                dataset["Insured_Amount2"],
                dataset["Insured_Amount3"]
                ]).T
    
    #將保險代碼轉為虛擬變數納入回歸資料集
    dummies = pd.get_dummies(input_dataset["Insurance_Coverage"])
    data_X = data_X.join(dummies)
    
    if input_dataset == policy_claim_test:
        list1 = pd.get_dummies(policy_claim_test["Insurance_Coverage"]).columns.tolist()
        list2 = pd.get_dummies(policy_claim_train["Insurance_Coverage"]).columns.tolist()
        list3 = [x for x in list2 if x not in list1]
        for i in list3:
            data_X[i] = 0
        
    return data_X, data_Y

## 進行 claim_policy_training 的資料整理的 function
def create_dataset_claim(input_dataset):
    dataset = input_dataset.copy()
    data_Y = df["Renewal"]
    #basic
    #dataset["車責"]
    #dataset["男"]
    #dataset["未婚"]
    data_X = pd.DataFrame([
                dataset["Premium"],
                dataset["Last_Renewal"],
                dataset["竊盜"],
                dataset["車損"],
                dataset["女"],
                dataset["法人"],
                dataset["已婚"],
                dataset["Dummy_TmNewa"],
                dataset["lia_class"],
                dataset["Engine_Displacement_(Cubic_Centimeter)"],
                dataset["Manafactured_Year_and_Month"],
                dataset["age"],
                dataset["Nature_of_the_claim"],
                dataset["Driver's_Gender"],
                dataset["Driver's_Relationship_with_Insured_1"],
                dataset["DOB_of_Driver"],
                dataset["Marital_Status_of_Driver"],
                dataset["Cause_of_Loss_1"],
                dataset["Paid_Loss_Amount"],
                dataset["paid_Expenses_Amount"],
                dataset["Salvage_or_Subrogation_1"],
                dataset["number_of_claimants"]]).T
    
    dummies = pd.get_dummies(input_dataset["Insurance_Coverage"])
    data_X.join(dummies)
    
    return data_X, data_Y

## 先將模型訓練出來
# 以 policy_claim_train 做模型
# 利用 logit model 做是否續約的預測
# 利用 linear model 做續約金額的預測

# logit model
policy_claim_train_after_preprocessing = data_preprocessing_policy(policy_claim_train)    
train_data_X, train_data_Y = create_dataset_policy(policy_claim_train_after_preprocessing)
logit = LogisticRegression()
logit_result = logit.fit(train_data_X, train_data_Y)

# linear model
linear_Y = policy_claim_train_after_preprocessing["Next_Premium"]
linear_X = policy_claim_train_after_preprocessing[["Last_Renewal",
                                                   "竊盜", "車損", 
                                                   "女", "法人", 
                                                   "國產車",
                                                   "lia_class", "plia_acc", "plia_acc", 
                                                   "claim", 
                                                   "Engine_Displacement_(Cubic_Centimeter)",
                                                   "Manafactured_Year_and_Month",
                                                   "age",
                                                   "Replacement_cost_of_insured_vehicle",
                                                   "Coverage_Deductible_if_applied",
                                                   "Insured_Amount2",
                                                   "Insured_Amount3"]]
dummies = pd.get_dummies(policy_claim_train["Insurance_Coverage"])
linear_X = linear_X.join(dummies)

linear = LinearRegression()
linear_result = linear.fit(linear_X, linear_Y)

## 利用模型進行預測
# 將 policy_test 丟入上述模型做預測

# logit model 的 data 創造
policy_claim_test_after_preprocessing = data_preprocessing_policy(policy_claim_test)
logit_predict_X, logit_predict_Y = create_dataset_policy(policy_claim_test_after_preprocessing)

# linear model 的 data 創造
linear_predict_X = policy_claim_test_after_preprocessing[["Last_Renewal",
                                                          "竊盜", "車損", 
                                                          "女", "法人",
                                                          "國產車",
                                                          "lia_class", "plia_acc", "plia_acc", 
                                                          "claim", 
                                                          "Engine_Displacement_(Cubic_Centimeter)",
                                                          "Manafactured_Year_and_Month",
                                                          "age",
                                                          "Replacement_cost_of_insured_vehicle",
                                                          "Coverage_Deductible_if_applied",
                                                          "Insured_Amount2",
                                                          "Insured_Amount3"]]
dummies = pd.get_dummies(policy_claim_test["Insurance_Coverage"])
linear_predict_X = linear_predict_X.join(dummies)


logit_prediction = logit.predict(logit_predict_X)
linear_prediction = linear.predict(linear_predict_X).round()

# join Renewal result
tmp = linear_predict_X.join(pd.Series(logit_prediction, name="Renewal"))
tmp["Renewal"].astype("int64")

# join Next_Premium result
tmp = tmp.join(pd.Series(linear_prediction, name="Next_Premium"))
tmp["Next_Premium"].astype("float64")

# join Policy_Number
tmp = tmp.join(policy_claim_test_after_preprocessing["Policy_Number"])

#replace Next_Premium = 0 if Renewal = 0
tmp.loc[tmp["Renewal"]==0, "Next_Premium"] = 0
tmp.loc[tmp["Next_Premium"]<0, "Next_Premium"] = 0

# 根據 Policy_Number 對 Next_Premium 做加總
tmp = tmp.groupby(by="Policy_Number").sum().reset_index()[["Policy_Number", "Next_Premium"]]

# 做出提交版本
submit = pd.merge(test["Policy_Number"].to_frame(), tmp, how="left", on="Policy_Number")
submit.to_csv("/Users/harry/Desktop/testing-set.csv", index=False)

## 進行回測
# 利用上面創造的 data_X, data_Y 進行回測

# logit 回測

# 創造資料
#policy_claim_train_after_preprocessing = data_preprocessing_policy(policy_claim_train)    
logit_data_X, logit_data_Y = create_dataset_policy(policy_claim_train_after_preprocessing)

# 切分資料
train_X, test_X, train_Y, test_Y = train_test_split(logit_data_X.drop("lia_class", axis=1), logit_data_Y, test_size = 0.3)

# 跑回歸
logit = LogisticRegression()
logit_result = logit.fit(train_X, train_Y)
test_Y_pred = logit.predict(test_X)

# 計算準確率
accuracy = logit.score(test_X, test_Y)
print("logit accuracy:", accuracy) #logit accuracy: 0.8006847916568502


logit_model = sm.Logit(logit_data_Y, logit_data_X)
result = logit_model.fit()
print(result.summary())

# Optimization terminated successfully.
#          Current function value: 0.485042
#          Iterations 6
#                            Logit Regression Results                           
# ==============================================================================
# Dep. Variable:                Renewal   No. Observations:              1061148
# Model:                          Logit   Df Residuals:                  1061132
# Method:                           MLE   Df Model:                           15
# Date:                Sun, 09 Sep 2018   Pseudo R-squ.:                 0.02934
# Time:                        16:59:28   Log-Likelihood:            -5.1470e+05
# converged:                       True   LL-Null:                   -5.3026e+05
#                                         LLR p-value:                     0.000
# ==========================================================================================================
#                                              coef    std err          z      P>|z|      [0.025      0.975]
# ----------------------------------------------------------------------------------------------------------
# Premium                                 4.943e-06   7.22e-07      6.851      0.000    3.53e-06    6.36e-06
# Last_Renewal                               0.5668      0.006     94.339      0.000       0.555       0.579
# 竊盜                                         0.0745      0.009      8.378      0.000       0.057       0.092
# 車損                                         0.0010      0.007      0.139      0.890      -0.013       0.015
# 女                                          0.1041      0.006     18.878      0.000       0.093       0.115
# 法人                                        -0.0915      0.008    -12.110      0.000      -0.106      -0.077
# 已婚                                         0.3799      0.006     68.951      0.000       0.369       0.391
# Dummy_TmNewa                               0.0549      0.009      6.447      0.000       0.038       0.072
# lia_class                                  0.0096      0.002      4.143      0.000       0.005       0.014
# plia_acc                                  -0.5879      0.022    -26.516      0.000      -0.631      -0.544
# pdmg_acc                                  -0.3768      0.012    -31.845      0.000      -0.400      -0.354
# claim                                     -0.1099      0.012     -8.960      0.000      -0.134      -0.086
# Engine_Displacement_(Cubic_Centimeter)     0.0002   5.15e-06     39.105      0.000       0.000       0.000
# Manafactured_Year_and_Month               -0.0485      0.001    -96.358      0.000      -0.049      -0.047
# age                                        0.0121      0.000     54.047      0.000       0.012       0.013
# Replacement_cost_of_insured_vehicle       -0.0013   3.44e-05    -36.983      0.000      -0.001      -0.001
# ==========================================================================================================

#policy_claim_train_after_preprocessing = data_preprocessing_policy(policy_claim_train)
linear_Y = policy_claim_train_after_preprocessing["Next_Premium"]
linear_X = policy_claim_train_after_preprocessing[["Last_Renewal",
                                                   "竊盜", "車損", 
                                                   "女", "法人",
                                                   "國產車",
                                                   "lia_class", "plia_acc", "plia_acc", 
                                                   "claim", 
                                                   "Engine_Displacement_(Cubic_Centimeter)",
                                                   "Manafactured_Year_and_Month",
                                                   "age",
                                                   "Replacement_cost_of_insured_vehicle",
                                                   "Coverage_Deductible_if_applied",
                                                   "Insured_Amount2",
                                                   "Insured_Amount3"]]
linear_model = sm.OLS(linear_Y, linear_X)
result = linear_model.fit()
print(result.summary())

#                             OLS Regression Results                            
# ==============================================================================
# Dep. Variable:           Next_Premium   R-squared:                       0.541
# Model:                            OLS   Adj. R-squared:                  0.541
# Method:                 Least Squares   F-statistic:                 7.807e+04
# Date:                Sun, 09 Sep 2018   Prob (F-statistic):               0.00
# Time:                        17:16:27   Log-Likelihood:            -1.1010e+07
# No. Observations:             1061148   AIC:                         2.202e+07
# Df Residuals:                 1061132   BIC:                         2.202e+07
# Df Model:                          16                                         
# Covariance Type:            nonrobust                                         
# ==========================================================================================================
#                                              coef    std err          t      P>|t|      [0.025      0.975]
# ----------------------------------------------------------------------------------------------------------
# Last_Renewal                             609.2955     19.734     30.875      0.000     570.617     647.974
# 竊盜                                      2243.8455     26.156     85.786      0.000    2192.580    2295.111
# 車損                                      2297.3401     20.845    110.210      0.000    2256.484    2338.196
# 女                                        123.4629     16.770      7.362      0.000      90.594     156.332
# 法人                                      3363.6495     22.181    151.644      0.000    3320.175    3407.124
# 國產車                                     -543.1783     20.418    -26.603      0.000    -583.197    -503.160
# lia_class                                499.0975      7.661     65.144      0.000     484.081     514.114
# plia_acc                                -265.8675     38.484     -6.909      0.000    -341.294    -190.441
# plia_acc                                -265.8675     38.484     -6.909      0.000    -341.294    -190.441
# claim                                   2078.1579     37.916     54.809      0.000    2003.843    2152.473
# Engine_Displacement_(Cubic_Centimeter)     0.7262      0.016     45.592      0.000       0.695       0.757
# Manafactured_Year_and_Month             -401.7594      1.534   -261.933      0.000    -404.766    -398.753
# age                                       73.0725      0.675    108.229      0.000      71.749      74.396
# Replacement_cost_of_insured_vehicle       31.0949      0.125    248.990      0.000      30.850      31.340
# Coverage_Deductible_if_applied             0.1523      0.010     14.978      0.000       0.132       0.172
# Insured_Amount2                            0.0003   6.35e-06     50.025      0.000       0.000       0.000
# Insured_Amount3                         4.175e-05   7.38e-07     56.569      0.000    4.03e-05    4.32e-05
# ==============================================================================
# Omnibus:                   751147.487   Durbin-Watson:                   0.287
# Prob(Omnibus):                  0.000   Jarque-Bera (JB):         55080361.396
# Skew:                           2.734   Prob(JB):                         0.00
# Kurtosis:                      37.869   Cond. No.                     5.41e+23
# ==============================================================================

# Warnings:
# [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
# [2] The smallest eigenvalue is 5.24e-28. This might indicate that there are
# strong multicollinearity problems or that the design matrix is singular.
