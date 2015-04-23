from django.db import models

class User(models.Model):
    RACE_CHOICES = (
            (1, 'White, Not Hispanic or Latino'),
            (2, 'Hispanic or Latino'),
            (3, 'Black or African American'),
            (4, 'American Indian and Alaska Native'),
            (5, 'Asian'),
            (6, 'Native Hawaiian and Other Pacific Islander'),
            (7, 'Some Other Race'),
            (8, 'Two or More Races'),
            )
    SEX_CHOICES = (
            (0, 'M'),
            (1, 'F'),
            )

#    tid = models.IntegerField(primary_key=True)
    uid = models.IntegerField(primary_key=True)
    zipcode = models.CharField(max_length=10)
    sex = models.IntegerField(choices=SEX_CHOICES)
    race = models.IntegerField(choices=RACE_CHOICES)
    income = models.IntegerField()

    def __str__(self):
        return str(self.uid)


class Store(models.Model):
    zipcode = models.CharField(max_length=10)
    latitude = models.FloatField()
    longitude = models.FloatField()



#class Transaction(models.Model):
#    uid = models.ForeignKey('User')
#    price = models.FloatField()
