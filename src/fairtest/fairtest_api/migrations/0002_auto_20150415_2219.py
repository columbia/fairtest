# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('fairtest_api', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='user',
            name='sex',
            field=models.IntegerField(choices=[(1, 'M'), (2, 'F')]),
            preserve_default=True,
        ),
    ]
