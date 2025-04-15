from Remote_User.models import (
    ClientRegister_Model,
    Tweet_Message_model,
    Tweet_Prediction_model,
    detection_ratio_model,
    detection_accuracy_model,
)
from django.shortcuts import render, redirect
from django.db.models import Count
import pandas as pd
import numpy as np
import re
import warnings
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
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

warnings.filterwarnings("ignore")

# ==========================
# User Authentication Views
# ==========================


def login(request):
    if request.method == "POST" and "submit1" in request.POST:
        username = request.POST.get("username")
        password = request.POST.get("password")
        try:
            enter = ClientRegister_Model.objects.get(
                username=username, password=password
            )
            request.session["userid"] = enter.id
            return redirect("Search_DataSets")
        except:
            pass
    return render(request, "RUser/login.html")


def Add_DataSet_Details(request):

    return render(request, "RUser/Add_DataSet_Details.html", {"excel_data": ""})


def Register1(request):
    if request.method == "POST":
        ClientRegister_Model.objects.create(
            username=request.POST.get("username"),
            email=request.POST.get("email"),
            password=request.POST.get("password"),
            phoneno=request.POST.get("phoneno"),
            country=request.POST.get("country"),
            state=request.POST.get("state"),
            city=request.POST.get("city"),
        )
        return render(request, "RUser/Register1.html")
    return render(request, "RUser/Register1.html")


def ViewYourProfile(request):
    obj = ClientRegister_Model.objects.get(id=request.session["userid"])
    return render(request, "RUser/ViewYourProfile.html", {"object": obj})


# ==========================
# Data Processing & Model Training
# ==========================


def Search_DataSets(request):
    if request.method == "POST":
        Tweet_Message = request.POST.get("keyword")
        df = pd.read_csv("./train_tweets.csv")
        df["processed_tweets"] = df["tweet"].apply(
            lambda tweet: " ".join(
                re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z 	])", " ", tweet.lower()).split()
            )
        )

        # Handle Imbalance
        df_class_fraud = df[df["label"] == 1]
        df_class_nonfraud = df[df["label"] == 0]
        df_class_fraud_oversample = df_class_fraud.sample(
            df_class_nonfraud.shape[0], replace=True
        )
        df_oversampled = pd.concat(
            [df_class_nonfraud, df_class_fraud_oversample], axis=0
        )

        X = df_oversampled["processed_tweets"]
        y = df_oversampled["label"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        count_vect = CountVectorizer(stop_words="english")
        transformer = TfidfTransformer(norm="l2", sublinear_tf=True)
        x_train_tfidf = transformer.fit_transform(count_vect.fit_transform(X_train))
        x_test_tfidf = transformer.transform(count_vect.transform(X_test))

        models = []
        svm_model = svm.LinearSVC().fit(x_train_tfidf, y_train)
        logreg_model = LogisticRegression().fit(x_train_tfidf, y_train)
        nb_model = MultinomialNB().fit(x_train_tfidf, y_train)
        models.append(("svm", svm_model))
        models.append(("logreg", logreg_model))
        models.append(("nb", nb_model))

        # ==========================
        # CNN Model
        # ==========================
        MAX_WORDS = 1000
        MAX_LEN = 100

        tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
        tokenizer.fit_on_texts(X_train)
        X_train_padded = pad_sequences(
            tokenizer.texts_to_sequences(X_train), maxlen=MAX_LEN
        )
        X_test_padded = pad_sequences(
            tokenizer.texts_to_sequences(X_test), maxlen=MAX_LEN
        )

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
        cnn_model.fit(
            X_train_padded,
            y_train,
            epochs=1,
            batch_size=32,
            validation_data=(X_test_padded, y_test),
        )

        class HybridVotingClassifier:
            def __init__(self, models, cnn_model, tokenizer, max_len):
                self.models = models
                self.cnn_model = cnn_model
                self.tokenizer = tokenizer
                self.max_len = max_len

            def predict(self, X):
                vectorized_X = count_vect.transform(X).toarray()
                preds = [model.predict(vectorized_X) for _, model in self.models]
                X_padded = pad_sequences(
                    self.tokenizer.texts_to_sequences(X), maxlen=self.max_len
                )
                cnn_preds = (
                    (self.cnn_model.predict(X_padded) > 0.5).astype("int32").flatten()
                )
                final_preds = np.round(np.mean(np.array(preds + [cnn_preds]), axis=0))
                return final_preds

        hybrid_model = HybridVotingClassifier(models, cnn_model, tokenizer, MAX_LEN)

        vectorized_tweet = [Tweet_Message]
        predict_svm = svm_model.predict(
            count_vect.transform(vectorized_tweet).toarray()
        )[0]
        predict_logreg = logreg_model.predict(
            count_vect.transform(vectorized_tweet).toarray()
        )[0]
        predict_nb = nb_model.predict(count_vect.transform(vectorized_tweet).toarray())[
            0
        ]
        predict_cnn = (
            (
                cnn_model.predict(
                    pad_sequences(
                        tokenizer.texts_to_sequences(vectorized_tweet),
                        maxlen=MAX_LEN,
                    )
                )
                > 0.5
            )
            .astype("int32")
            .flatten()[0]
        )

        predict_voting = hybrid_model.predict(vectorized_tweet)[0]

        label_map = {
            0: "Non Offensive or Non Cyberbullying",
            1: "Offensive or Cyberbullying",
        }

        svm_result = label_map[predict_svm]
        logreg_result = label_map[predict_logreg]
        nb_result = label_map[predict_nb]
        cnn_result = label_map[predict_cnn]
        final_result = label_map[predict_voting]

        Tweet_Prediction_model.objects.create(
            Tweet_Message=Tweet_Message, Prediction_Type=final_result
        )

        return render(
            request,
            "RUser/Search_DataSets.html",
            {
                "objs": final_result,
            },
        )

    return render(request, "RUser/Search_DataSets.html")
