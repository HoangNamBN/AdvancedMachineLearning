import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import learning_curve
from sklearn import svm
from sklearn.decomposition import PCA

stopword = []
stopword.append('.')
stopword.append(',')
stopword.append(';')

df = pd.read_excel('D:\EPUIT\Learn\MachineLearning\Advanced\BaiTapThucHanh\Buoi1\Data\sentimentvn.xlsx')

X = df['text'].to_list()
y = df['label'].to_list()
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=20, random_state=10)

Vectorizer = CountVectorizer(analyzer = u'word', max_df = 0.95, ngram_range=(1,1), stop_words=set(stopword))
TfIdf = Vectorizer.fit(X_train)
X_train = TfIdf.transform(X_train)
pca = PCA(n_components=1500)
X_train= pca.fit_transform(X_train.todense())
model = svm.LinearSVC(C=0.1)
model.fit(X_train,y_train)
X_test = TfIdf.transform(X_test)
X_test = pca.transform(X_test.todense())
y_pred = model.predict(X_test)
print(classification_report(y_test,y_pred))