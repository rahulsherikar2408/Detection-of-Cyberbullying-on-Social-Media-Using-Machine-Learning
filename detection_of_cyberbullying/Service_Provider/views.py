from django.db.models import Count, Avg
from django.shortcuts import render, redirect
from django.db.models import Count
from django.db.models import Q
import xlwt
from django.http import HttpResponse
import warnings
import json
from django.shortcuts import render
from django.db.models import Avg
from django.core.serializers.json import DjangoJSONEncoder
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import re


# Create your views here.
from Remote_User.models import (
    ClientRegister_Model,
    Tweet_Message_model,
    Tweet_Prediction_model,
    detection_ratio_model,
    detection_accuracy_model,
)


# Create your views here.


def serviceproviderlogin(request):
    if request.method == "POST":
        admin = request.POST.get("username")
        password = request.POST.get("password")
        if admin == "Admin" and password == "Admin":
            detection_accuracy_model.objects.all().delete()
            return redirect("View_Remote_Users")

    return render(request, "SProvider/serviceproviderlogin.html")


def Find_Cyberbullying_Prediction_Ratio(request):
    detection_ratio_model.objects.all().delete()
    ratio = ""
    kword = "Non Offensive or Non Cyberbullying"
    print(kword)
    obj = Tweet_Prediction_model.objects.all().filter(Q(Prediction_Type=kword))
    obj1 = Tweet_Prediction_model.objects.all()
    count = obj.count()
    count1 = obj1.count()
    ratio = (count / count1) * 100
    if ratio != 0:
        detection_ratio_model.objects.create(names=kword, ratio=ratio)

    ratio1 = ""
    kword1 = "Offensive or Cyberbullying"
    print(kword1)
    obj1 = Tweet_Prediction_model.objects.all().filter(Q(Prediction_Type=kword1))
    obj11 = Tweet_Prediction_model.objects.all()
    count1 = obj1.count()
    count11 = obj11.count()
    ratio1 = (count1 / count11) * 100
    if ratio1 != 0:
        detection_ratio_model.objects.create(names=kword1, ratio=ratio1)

    obj = detection_ratio_model.objects.all()
    return render(
        request, "SProvider/Find_Cyberbullying_Prediction_Ratio.html", {"objs": obj}
    )


def View_Remote_Users(request):
    obj = ClientRegister_Model.objects.all()
    return render(request, "SProvider/View_Remote_Users.html", {"objects": obj})


def ViewTrendings(request):
    topic = (
        Tweet_Prediction_model.objects.values("topics")
        .annotate(dcount=Count("topics"))
        .order_by("-dcount")
    )
    return render(request, "SProvider/ViewTrendings.html", {"objects": topic})


def charts(request, chart_type):
    chart = detection_ratio_model.objects.values("names").annotate(
        dcount=Avg("ratio")
    )
    chart_data = list(chart)  # Convert queryset to list of dicts

    return render(
        request,
        "SProvider/charts.html",
        {
            "chart_type": chart_type,
            "chart_data": json.dumps(chart_data, cls=DjangoJSONEncoder),
        },
    )


def charts1(request, chart_type):
    # Aggregate accuracy by model name
    charts = detection_accuracy_model.objects.values("names").annotate(
        dcount=Avg("ratio")
    )

    chart_data = list(charts)  # Convert QuerySet to list of dictionaries

    return render(request, "SProvider/charts1.html", {
        "chart_type": chart_type,
        "chart_data": json.dumps(chart_data, cls=DjangoJSONEncoder),
    })


def View_Cyberbullying_Predict_Type(request):

    obj = Tweet_Prediction_model.objects.all()
    return render(
        request, "SProvider/View_Cyberbullying_Predict_Type.html", {"list_objects": obj}
    )


def likeschart(request, like_chart):
    charts = detection_accuracy_model.objects.values("names").annotate(
        dcount=Avg("ratio")
    )
    chart_data = list(charts)  # Convert queryset to list of dicts

    return render(
        request,
        "SProvider/likeschart.html",
        {
            "like_chart": like_chart,
            "chart_data": json.dumps(chart_data, cls=DjangoJSONEncoder),
        },
    )


def Download_Cyber_Bullying_Prediction(request):

    response = HttpResponse(content_type="application/ms-excel")
    # decide file name
    response["Content-Disposition"] = (
        'attachment; filename="Cyberbullying_Predicted_DataSets.xls"'
    )
    # creating workbook
    wb = xlwt.Workbook(encoding="utf-8")
    # adding sheet
    ws = wb.add_sheet("sheet1")
    # Sheet header, first row
    row_num = 0
    font_style = xlwt.XFStyle()
    # headers are bold
    font_style.font.bold = True
    # writer = csv.writer(response)
    obj = Tweet_Prediction_model.objects.all()
    data = obj  # dummy method to fetch data.
    for my_row in data:
        row_num = row_num + 1
        ws.write(row_num, 0, my_row.Tweet_Message, font_style)
        ws.write(row_num, 1, my_row.Prediction_Type, font_style)
    wb.save(response)
    return response


def train_model(request):
    detection_accuracy_model.objects.all().delete()

    df = pd.read_csv("./train_tweets.csv")
    df.head()
    offensive_tweet = df[df.label == 1]
    offensive_tweet.head()
    normal_tweet = df[df.label == 0]
    normal_tweet.head()
    # Offensive Word clouds
    from os import path
    from PIL import Image
    from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

    text = " ".join(review for review in offensive_tweet)
    wordcloud = WordCloud(
        max_font_size=50, max_words=100, background_color="white"
    ).generate(text)
    fig = plt.figure(figsize=(20, 6))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    # plt.show()
    # distributions
    df_Stat = df[["label", "tweet"]].groupby("label").count().reset_index()
    df_Stat.columns = ["label", "count"]
    df_Stat["percentage"] = (df_Stat["count"] / df_Stat["count"].sum()) * 100
    df_Stat

    def process_tweet(tweet):
        return " ".join(
            re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])", " ", tweet.lower()).split()
        )

    df["processed_tweets"] = df["tweet"].apply(process_tweet)
    df.head()
    # As this dataset is highly imbalance we have to balance this by over sampling
    cnt_non_fraud = df[df["label"] == 0]["processed_tweets"].count()
    df_class_fraud = df[df["label"] == 1]
    df_class_nonfraud = df[df["label"] == 0]
    df_class_fraud_oversample = df_class_fraud.sample(cnt_non_fraud, replace=True)
    df_oversampled = pd.concat([df_class_nonfraud, df_class_fraud_oversample], axis=0)

    print("Random over-sampling:")
    print(df_oversampled["label"].value_counts())
    # Split data into training and test sets
    from sklearn.model_selection import train_test_split

    X = df_oversampled["processed_tweets"]
    y = df_oversampled["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=None
    )
    from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

    count_vect = CountVectorizer(stop_words="english")
    transformer = TfidfTransformer(norm="l2", sublinear_tf=True)
    x_train_counts = count_vect.fit_transform(X_train)
    x_train_tfidf = transformer.fit_transform(x_train_counts)
    print(x_train_counts.shape)
    print(x_train_tfidf.shape)
    x_test_counts = count_vect.transform(X_test)
    x_test_tfidf = transformer.transform(x_test_counts)

    # =============================================================================
    # SVM Model
    from sklearn import svm
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

    lin_clf = svm.LinearSVC()
    lin_clf.fit(x_train_tfidf, y_train)
    predict_svm = lin_clf.predict(x_test_tfidf)
    svm_acc = accuracy_score(y_test, predict_svm) * 100
    print("SVM ACCURACY")
    print(svm_acc)
    detection_accuracy_model.objects.create(names="SVM", ratio=svm_acc)

    print(confusion_matrix(y_test, predict_svm))
    print(classification_report(y_test, predict_svm))

    # =============================================================================
    # Logistic Regression Model

    from sklearn.linear_model import LogisticRegression

    logreg = LogisticRegression(random_state=42)

    # Building Logistic Regression  Model
    logreg.fit(x_train_tfidf, y_train)
    predict_log = logreg.predict(x_test_tfidf)
    logistic = accuracy_score(y_test, predict_log) * 100
    print("Logistic Accuracy")
    print(logistic)
    detection_accuracy_model.objects.create(names="Logistic Regression", ratio=logistic)

    print(confusion_matrix(y_test, predict_log))
    print(classification_report(y_test, predict_log))

    # =============================================================================
    # Naive Bayes model

    from sklearn.naive_bayes import MultinomialNB

    NB = MultinomialNB()
    NB.fit(x_train_tfidf, y_train)
    predict_nb = NB.predict(x_test_tfidf)
    naivebayes = accuracy_score(y_test, predict_nb) * 100
    print("Naive Bayes")
    print(naivebayes)
    detection_accuracy_model.objects.create(names="Naive Bayes", ratio=naivebayes)

    print(confusion_matrix(y_test, predict_nb))
    print(classification_report(y_test, predict_nb))

    # =============================================================================
    # CNN Model

    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (
        Embedding,
        Conv1D,
        GlobalMaxPooling1D,
        Dense,
        Dropout,
    )
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    # Data Preprocessing for CNN
    MAX_WORDS = 2000  # Vocabulary size
    MAX_LEN = 100  # Max length of input sequences

    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(df_oversampled["processed_tweets"])

    X_sequences = tokenizer.texts_to_sequences(df_oversampled["processed_tweets"])
    X_padded = pad_sequences(
        X_sequences, maxlen=MAX_LEN, padding="post", truncating="post"
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_padded, y, test_size=0.2, stratify=None
    )
    # No need to reshape manually, keep it as (batch_size, MAX_LEN)
    print(X_train.shape)  # Expected: (47552, 100)

    # Define CNN Model
    cnn_model = Sequential(
        [
            Embedding(MAX_WORDS, 128, input_length=MAX_LEN),
            Conv1D(64, 5, activation="relu"),
            GlobalMaxPooling1D(),
            Dense(64, activation="relu"),
            Dropout(0.5),
            Dense(1, activation="sigmoid"),
        ]
    )

    cnn_model.compile(
        loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
    )

    # Train Model
    cnn_model.fit(
        X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test)
    )

    # CNN Model Evaluation
    y_pred_cnn = (cnn_model.predict(X_test) > 0.5).astype("int32")
    cnn_acc = accuracy_score(y_test, y_pred_cnn) * 100
    print("CNN ACCURACY")
    print(cnn_acc)
    detection_accuracy_model.objects.create(names="CNN", ratio=cnn_acc)

    from sklearn.metrics import confusion_matrix, classification_report

    print(confusion_matrix(y_test, y_pred_cnn))
    print(classification_report(y_test, y_pred_cnn))

    # Test Data Set
    df_test = pd.read_csv("./test_tweets.csv")
    df_test.head()
    df_test.shape
    df_test["processed_tweets"] = df_test["tweet"].apply(process_tweet)
    df_test.head()
    X = df_test["processed_tweets"]
    x_test_counts = count_vect.transform(X)
    x_test_tfidf = transformer.transform(x_test_counts)
    df_test["predict_nb"] = NB.predict(x_test_tfidf)
    df_test[df_test["predict_nb"] == 1]
    df_test["predict_svm"] = NB.predict(x_test_tfidf)

    X_sequences_test = tokenizer.texts_to_sequences(X)
    X_padded_test = pad_sequences(
        X_sequences_test, maxlen=MAX_LEN, padding="post", truncating="post"
    )
    df_test["predict_cnn"] = (cnn_model.predict(X_padded_test) > 0.5).astype("int32")

    df_test.head()
    file_name = "Predictions.csv"
    df_test.to_csv(file_name, index=False)

    obj = detection_accuracy_model.objects.all()
    return render(
        request,
        "SProvider/train_model.html",
        {
            "objs": obj,
            "svmcm": confusion_matrix(y_test, predict_svm),
            "lrcm": confusion_matrix(y_test, predict_log),
            "nbcm": confusion_matrix(y_test, predict_nb),
            "cnncm": confusion_matrix(y_test, y_pred_cnn),
        },
    )
