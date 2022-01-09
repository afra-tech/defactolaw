from contracts.models import Contract
from django import forms

class DocumentForm(forms.ModelForm):
    class Meta:
        model = Contract
        fields = ('pdf','scanned')