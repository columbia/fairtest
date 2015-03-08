from django.db import models


class Product(models.Model):
    """A product to be sold online

    Attributes:
        name(str): The name of the product
        basprice(decimal): The initial price of the product,  which is subject
                           to modifications based on the location of competitors
    """
    name = models.CharField(max_length=200)
    baseprice = models.DecimalField(max_digits=15,decimal_places=2,default=0)

    def __str__(self):
        return self.name

    def base_price(self):
        return self.baseprice


class Competitor(models.Model):
    """A Competitor of the selle.

    Attributes:
        name(str): The name of the competitor
        basprice(decimal): The address of the competitor.
                           to modifications based on the location of competitors

    Note:
        We need to set a specific format for the addresses.
    """
    name = models.CharField(max_length=200)
    address = models.CharField(max_length=200)

    def __str__(self):
        return self.name + self.address


class Zipcode(models.Model):
    """A Zipcode geocoding database.

    Attributes:
        zip(str): Zipcode
        city(str): City represented by the zipcode
        state(str): State represented by the zipcode
        la(float): Latitude of the zipcode
        lo(float): Longtitude of the zipcode
    """
    zip = models.CharField(max_length=6)
    city = models.CharField(max_length=50)
    state = models.CharField(max_length=3)

    la = models.FloatField()
    lo = models.FloatField()

    def __str__(self):
        return self.city + ", " + self.state + self.zip
