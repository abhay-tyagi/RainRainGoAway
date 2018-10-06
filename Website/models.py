from django.db import models

# Create your models here.

class Vid(models.Model):
	
	inputVid = models.FileField(upload_to='', null=True, blank=True)
	outputVid = models.FileField(upload_to='', null=True, blank=True)