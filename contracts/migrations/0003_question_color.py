# Generated by Django 3.2.5 on 2021-08-14 22:27

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('contracts', '0002_question'),
    ]

    operations = [
        migrations.AddField(
            model_name='question',
            name='color',
            field=models.CharField(blank=True, max_length=50),
        ),
    ]
