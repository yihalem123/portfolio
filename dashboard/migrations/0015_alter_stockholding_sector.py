# Generated by Django 3.2.3 on 2024-03-29 07:34

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('dashboard', '0014_stockholding_sector'),
    ]

    operations = [
        migrations.AlterField(
            model_name='stockholding',
            name='sector',
            field=models.CharField(default='Unknown', max_length=50),
        ),
    ]