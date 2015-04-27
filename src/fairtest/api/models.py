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
    INCOME_CHOICES = (
            (1, 'income < 5000$'),
            (2, '5000   <= income < 10000$'),
            (3, '10000  <= income < 20000'),
            (4, '20000  <= income < 40000'),
            (5, '40000  <= income < 80000'),
            (6, '80000  <= income < 160000'),
            (7, '160000 <= income < 320000'),
            (8, '320000 <= income '),
            )
    uid = models.IntegerField(primary_key=True)
    zipcode = models.CharField(max_length=10)
    sex = models.IntegerField(choices=SEX_CHOICES)
    race = models.IntegerField(choices=RACE_CHOICES)
    income = models.IntegerField(choices=INCOME_CHOICES)

    def __str__(self):
        return str(self.uid)

    def get_attribute(self, attribute):
        if attribute == "sex":
            return str(self.sex)
        elif attribute == "income":
            return str(self.income)
        elif attribute == "race":
            return str(self.race)
        else:
            return ""


class Store(models.Model):
    zipcode = models.CharField(primary_key=True, max_length=10)
    latitude = models.FloatField()
    longitude = models.FloatField()


class Competitor(models.Model):
    zipcode = models.CharField(primary_key=True, max_length=10)
    latitude = models.FloatField()
    longitude = models.FloatField()


class Zipcode(models.Model):
    zipcode = models.CharField(primary_key=True, max_length=10)
    latitude = models.FloatField()
    longitude = models.FloatField()
