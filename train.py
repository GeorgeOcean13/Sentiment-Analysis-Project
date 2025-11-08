import pandas as pd
import re 
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB       # In case if you wanna use Naive-Bayes together with logistic regression.
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score


# ==== Trying again because the model wasn't accurate ===

# STEP 1:
def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text



# ====== Loading the Dataset ========
print("====LOADING DATASET====")
df =  pd.read_csv("data/tweet_sentiment.csv")
df = df.dropna(subset=['text'])
# df.rename(columns={"review":"text"},inplace=True)
print(f"DATASET LOADED : {df.shape[0]} rows")

# This is Optional : If you want to test your model on a Smaller Sample
# df = df.sample(10000,random_state=42)

# STEP 2: Clean text column
print("Cleaning Text ...")
df['text'] = df['text'].apply(clean_text)

x = df['text']
y = df['sentiment']
# STEP 3: Train-Test Split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,stratify=y,random_state=42)
print(f"Training set: {x_train.shape[0]} rows")
print(f"Test set: {x_test.shape[0]} rows")

# Pipelining ===> Chain of steps into one object.
# ====> It automatically applies the vectorizer and classifier in order.

# STEP 4: Create Pipeline 
print('Building Pipeline ...')
model = Pipeline([
    ("tfidf",TfidfVectorizer(stop_words="english",max_df=0.95,min_df=2,ngram_range=(1,3),max_features=20000)),
    ("clf",LogisticRegression(C=1.0,max_iter=1000,class_weight='balanced'))
])

# STEP 5: Train the Model
print("Training Model")
model.fit(x_train,y_train)


# Step6: Evaluating the Model
y_pred = model.predict(x_test)
print("\n📊 Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

# Now, Save the Model
if joblib.dump(model,'models/sentiment_model2.pkl'):
    print("Your Model is Tested and Saved :)")
