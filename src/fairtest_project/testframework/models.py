from django.db import models

class Product(models.Model):
    name = models.CharField(max_length=200)
    baseprice = models.DecimalField(max_digits=15,decimal_places=2,default=0)

    def __str__(self):
        return self.name

    def base_price(self):
        return self.baseprice


class Competitor(models.Model):
    name = models.CharField(max_length=200)
    address = models.CharField(max_length=200)

    def __str__(self):
        return self.name + self.address
