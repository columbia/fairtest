# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Transaction',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('price', models.FloatField()),
            ],
            options={
            },
            bases=(models.Model,),
        ),
        migrations.CreateModel(
            name='User',
            fields=[
                ('id', models.IntegerField(primary_key=True, serialize=False)),
                ('zip', models.CharField(max_length=10)),
                ('sex', models.IntegerField()),
                ('race', models.IntegerField(choices=[(1, 'White, Not Hispanic or Latino'), (2, 'Hispanic or Latino'), (3, 'Black or African American'), (4, 'American Indian and Alaska Native'), (5, 'Asian'), (6, 'Native Hawaiian and Other Pacific Islander'), (7, 'Some Other Race'), (8, 'Two or More Races')])),
            ],
            options={
            },
            bases=(models.Model,),
        ),
        migrations.AddField(
            model_name='transaction',
            name='uid',
            field=models.ForeignKey(to='fairtest_api.User'),
            preserve_default=True,
        ),
    ]
