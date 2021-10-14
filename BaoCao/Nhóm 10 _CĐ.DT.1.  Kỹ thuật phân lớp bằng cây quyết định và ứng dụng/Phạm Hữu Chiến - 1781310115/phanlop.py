import pandas as pd
import numpy as np
#import pydotplus
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from IPython.display import Image
from sklearn import preprocessing
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pylab as pl
#Doc du lieu
df = pd.read_csv("soccer_international_history.csv")
#Tien xu ly du lieu
data = df[["home_country","away_country","home_score", "away_score", "match_type","match_city","match_country","home_team_result"]].copy(True)
data = data.dropna()
print("Du lieu:\n", data)
LanberEncoder = preprocessing.LabelEncoder()
data['home_country_replace'] = LanberEncoder.fit_transform(data['home_country'])
data['away_country_replace'] = LanberEncoder.fit_transform(data['away_country'])
data['match_type_replace'] = LanberEncoder.fit_transform(data['match_type'])
data['match_city_replace'] = LanberEncoder.fit_transform(data['match_city'])
data['match_country_replace'] = LanberEncoder.fit_transform(data['match_country'])
data['home_team_result_replace'] = LanberEncoder.fit_transform(data['home_team_result'])
data_train = data.drop(["home_country","away_country","match_city","match_type","match_country", "home_team_result"], axis=1)
print("Du lieu chuan hoa:\n", data_train.head(20))

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

#clf = SVC(gamma='auto')

X = np.asarray(data_train[['home_country_replace','away_country_replace','match_city_replace','home_score', 'away_score', 'match_type_replace','match_country_replace','home_team_result_replace']])
Y = np.asarray(data_train['home_team_result_replace'])
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
# clf.fit(X_train, Y_train)
# score = clf.score(X_test, Y_test)
# print("Score", score)
# Du doan voi tao du lieu train X_test , Y_test
decsionTreeEntropy = DecisionTreeClassifier(criterion="entropy", random_state=100, max_depth=5, min_samples_leaf=5)
decsionTreeEntropy.fit(X_train, Y_train)
y_preEntropy = decsionTreeEntropy.predict(X_test)
print("Du doan bang Entrophy",y_preEntropy)
print( "Kich co cua X_train", X_train.shape)
# gini = DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=5, min_samples_leaf=5)
# gini.fit(X_train, y_train)
# y_preGini =gini.predict(X_test)
#print("Du doan bang Gini",y_preGini)
#Bước 3: Đánh giá mô hình học dựa trên kết quả dự đoán (với độ đo đơn giản Accuracy, Precision, Recall)
print("Accuracy Score:", accuracy_score(Y_test, y_preEntropy)*100)
#classfier report
from sklearn.metrics import classification_report
print("Bao cao:\n", classification_report(Y_test, y_preEntropy))
#Confusion matrix
from sklearn.metrics import confusion_matrix
print('Ma tran hon loan:\n', confusion_matrix(Y_test, y_preEntropy))
# cm = confusion_matrix(Y_test, y_preEntropy)
# pl.matshow(cm)
# pl.title('Ma trận nhầm lẫn của bộ phân loại')
# pl.colorbar()
# pl.show()

#Bieu dien du lieu
#Giam chieu PCA
# pca = PCA(n_components=1)
# X_scat = pca.fit_transform(X_train)
#Plot
# plt.scatter(X_train,Y_train,s=100,c='green',edgecolors='black',)
# plt.xlabel("Giá trị X")
# plt.ylabel("Giá trị Y")
# plt.title("Biểu đồ tương quan X và Y")
# plt.show()
# fig = plt.figure(figsize=(10,7))
# colors=['red' if l==0 else 'blue' for l in Y_train]
# plt.scatter(X_train[:, 0], X_train[:, 1], label='Logistics regression', color=colors)
# plt.plot(X_train, Y_train, label='Decision Boundary')
# plt.show()
#Tao dot data
# dot_data = tree.export_graphviz(decsionTreeEntropy, out_file=None)
# # Ve do thi
# graph = pydotplus.graph_from_dot_data(dot_data)
# #Show do thi
# Image(graph.create_png)
# #Tao file PDF
# graph.write_pdf("train_titanic.pdf")
# #Tao anh PNG
# graph.write_png("train_titanic.png")
