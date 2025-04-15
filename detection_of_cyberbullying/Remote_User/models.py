from django.db import models

# Create your models here.
from django.db.models import CASCADE


class ClientRegister_Model(models.Model):
    username = models.CharField(max_length=30)
    email = models.EmailField(max_length=30)
    password = models.CharField(max_length=10)
    phoneno = models.CharField(max_length=10)
    country = models.CharField(max_length=30)
    state = models.CharField(max_length=30)
    city = models.CharField(max_length=30)

class Tweet_Message_model(models.Model):

    Tweet_Id=models.CharField(max_length=300)
    Tweet_Id=models.CharField(max_length=300)
    Tweet_Message=models.CharField(max_length=300)

class Tweet_Prediction_model(models.Model):


    Tweet_Message=models.CharField(max_length=300)
    Prediction_Type=models.CharField(max_length=300)

class detection_accuracy_model(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)

class detection_ratio_model(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)



