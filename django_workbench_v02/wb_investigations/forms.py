from django import forms

from .models import InvestigationStatus


class InvestigationCreateForm(forms.Form):
    title = forms.CharField(max_length=255)
    question_text = forms.CharField(required=False, widget=forms.Textarea)
    scope_json = forms.JSONField(required=False, initial=dict)
    method_json = forms.JSONField(required=False, initial=dict)
    status = forms.ChoiceField(
        choices=InvestigationStatus.choices,
        initial=InvestigationStatus.DRAFT,
    )


class InvestigationUpdateForm(forms.Form):
    title = forms.CharField(max_length=255)
    question_text = forms.CharField(required=False, widget=forms.Textarea)
    scope_json = forms.JSONField(required=False, initial=dict)
    method_json = forms.JSONField(required=False, initial=dict)
    status = forms.ChoiceField(choices=InvestigationStatus.choices)
