import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
candidates = pd.read_csv("export_dataframe.csv", header=0)
df = pd.DataFrame(candidates,columns= ['gmat',
'gpa','work_experience','admitted'])
X = df[['gmat', 'gpa','work_experience']]
y = df['admitted']
X_train,X_test,y_train,y_test =train_test_split(X,y,test_size=0.25,random_state=0) 
logistic_regression= LogisticRegression()
logistic_regression.fit(X_train,y_train)
new_candidates = {'gmat': [590,740,680,610,710],
 'gpa': [2,3.7,3.3,2.3,3],
 'work_experience': [3,4,6,1,5]
 }
df2 = pd.DataFrame(new_candidates,columns= ['gmat', 'gpa','work_experience'])
y_pred=logistic_regression.predict(df2)
print (df2)
print (y_pred)
