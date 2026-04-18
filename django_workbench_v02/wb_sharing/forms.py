from django import forms

from .models import ShareMode


class ShareLinkCreateForm(forms.Form):
    mode = forms.ChoiceField(choices=ShareMode.choices, initial=ShareMode.SNAPSHOT)
    is_public = forms.BooleanField(required=False, initial=True)
    expires_at = forms.DateTimeField(
        required=False,
        widget=forms.DateTimeInput(attrs={"type": "datetime-local"}),
        help_text="Optional expiry (UTC). Leave blank for no expiry.",
    )


class ShareLinkUpdateForm(forms.Form):
    mode = forms.ChoiceField(choices=ShareMode.choices)
    is_public = forms.BooleanField(required=False)
    is_active = forms.BooleanField(required=False)
    expires_at = forms.DateTimeField(
        required=False,
        widget=forms.DateTimeInput(attrs={"type": "datetime-local"}),
        help_text="Optional expiry (UTC). Leave blank for no expiry.",
    )
