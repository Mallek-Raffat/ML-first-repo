import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectPercentile, chi2, SelectKBest ,f_classif
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

df=pd.read_csv(r"D:\data analysis\DEPI\Excel\breast-cancer.csv")

df["diagnosis"]=df["diagnosis"].map({"M":1,"B":0})
X=df.drop(["diagnosis","id"],axis=1)
Y=df["diagnosis"]
SF=SelectPercentile(score_func=chi2,percentile=20)
X1=SF.fit_transform(X,Y)
selectedFeats=X.columns[SF.get_support()]
X=pd.DataFrame(X1,columns=selectedFeats)
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=.2,random_state=44)
accScore=[]
numberS=df.select_dtypes(include="number")
numberS=numberS.drop("id",axis=1)
sns.heatmap(X.corr(),annot=True)
plt.show()
def ALL(model):
    model.fit(X_train,Y_train)
    pred=model.predict(X_test)
    A=accuracy_score(Y_test,pred)
    accScore.append(A)



model1=RandomForestClassifier(random_state=44)
ALL(model1)
model2=LogisticRegression(random_state=44)
ALL(model2)
model3=DecisionTreeClassifier(random_state=44)
ALL(model3)
model4=GradientBoostingClassifier(random_state=44)
ALL(model4)
model5=KNeighborsClassifier()
ALL(model5)
model6=SVC(random_state=44)
ALL(model6)
model7=GaussianNB()
ALL(model7)


algors=["RandomForestClassifier","LogisticRegression","DecisionTreeClassifier","GradientBoostingClassifier","KNeighborsClassifier","SVC","GaussianNB"]
accuracies = list(zip(algors, accScore))
accuracies = pd.DataFrame(accuracies, columns=["algorithms", "accuracy"])
accuracies=accuracies.sort_values("accuracy",ascending=False)
print(accuracies)
