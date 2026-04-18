from django import forms

from .models import RunType


class RunQueueForm(forms.Form):
    run_type = forms.ChoiceField(choices=RunType.choices)
    input_config_json = forms.JSONField(required=False, initial=dict)
    query_start_date = forms.DateField(required=False, widget=forms.DateInput(attrs={"type": "date"}))
    query_end_date = forms.DateField(required=False, widget=forms.DateInput(attrs={"type": "date"}))


class RunCancelForm(forms.Form):
    cancel_reason = forms.CharField(required=False, widget=forms.Textarea)
