from django.core.exceptions import ValidationError
from django.db import models
from django.utils import timezone


# Model
class Contract(models.Model):

    def validate_file_extension(value):
        if value.file.content_type != 'application/pdf':
            raise ValidationError('Error message')

    pdf = models.FileField(validators=[validate_file_extension])
    name = models.CharField(max_length=100, default="output.pdf")
    created_on = models.DateTimeField(default=timezone.now)
    scanned = models.BooleanField(default=False)

    def __str__(self):
        return self.name
    def getfile(self):
        return self.cleaned_data['pdf']

    @classmethod
    def object(cls):
        return cls._default_manager.all().first()  # Since only one item

    def save(self, *args, **kwargs):
        self.pk = self.id = 1
        return super().save(*args, **kwargs)


class Question(models.Model):
    """Class for holding question objects"""

    ques = models.TextField(max_length=1000)
    description = models.TextField(max_length=1000)
    alias = models.TextField(max_length=100)
    color = models.CharField(max_length=50,blank=True)

    class Meta:
        """Meta definition for Question."""

        verbose_name = 'Question'
        verbose_name_plural = 'Questions'

    def __str__(self):
        return self.alias