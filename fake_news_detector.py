import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#load datasets

fake_news = pd.read_csv("dataset/Fake.csv")
true_news = pd.read_csv("dataset/True.csv")

# Add Labels
fake_news['label']=0
true_news['label']=1

#combine datasets
data = pd.concat([fake_news,true_news])
#shuffle data
data = data.sample(frac=1)
#use only text column
data = data[['text','label']]

#Split data
X_train,X_test,y_train,y_test = train_test_split(
    data['text'],
    data['label'],
    test_size=0.2
)
# Vectorization
vectorizer = TfidfVectorizer(stop_words='english',max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
#model
model = LogisticRegression()
model.fit(X_train_vec,y_train)
#prediction
pred = model.predict(X_test_vec)
print(" Accuracy:",accuracy_score(y_test,pred))
user_news = input("Enter custom news : ")
news = [user_news]
news_vec = vectorizer.transform(news)
result =  model.predict(news_vec)
if result==1:
  news_msg = "This news is Real"
else:
  news_msg = "This news is Fake"
print("Prediction: ",news_msg)