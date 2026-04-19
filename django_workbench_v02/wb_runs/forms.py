from django import forms

from wb_notifications.models import NotificationTrigger

from .models import RunType


class RunQueueForm(forms.Form):
    run_type = forms.ChoiceField(choices=RunType.choices)
    input_config_json = forms.JSONField(required=False, initial=dict)
    query_start_date = forms.DateField(required=False, widget=forms.DateInput(attrs={"type": "date"}))
    query_end_date = forms.DateField(required=False, widget=forms.DateInput(attrs={"type": "date"}))
    request_completion_email = forms.BooleanField(
        required=False,
        initial=False,
        help_text="Email me when this run reaches a terminal status.",
    )
    notify_on = forms.ChoiceField(
        required=False,
        choices=NotificationTrigger.choices,
        initial=NotificationTrigger.ANY,
        help_text="When to send email if completion notification is enabled.",
    )


class RunCancelForm(forms.Form):
    cancel_reason = forms.CharField(required=False, widget=forms.Textarea)
