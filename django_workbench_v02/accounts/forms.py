from django import forms
from django.contrib.auth.forms import ReadOnlyPasswordHashField
from django.core.validators import validate_email

from .models import User


class ServicesEnquiryForm(forms.Form):
    WORK_TYPE_CHOICES = (
        ("Rapid scoping review", "Rapid scoping review"),
        ("Full custom analysis", "Full custom analysis"),
        ("Dedicated dashboard", "Dedicated dashboard"),
        ("Publication or grant support", "Publication or grant support"),
    )

    name = forms.CharField(max_length=120)
    organisation = forms.CharField(max_length=180, required=False)
    email = forms.EmailField(max_length=254)
    work_type = forms.ChoiceField(choices=WORK_TYPE_CHOICES)
    project_summary = forms.CharField(min_length=20, max_length=4000, widget=forms.Textarea)
    website = forms.CharField(required=False)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_spam = False

    def clean_website(self):
        value = str(self.cleaned_data.get("website") or "").strip()
        if value:
            self.is_spam = True
        return value

    def clean_email(self):
        value = str(self.cleaned_data.get("email") or "").strip()
        validate_email(value)
        return value.lower()

    def email_subject(self) -> str:
        work_type = self.cleaned_data.get("work_type") or "Services enquiry"
        organisation = self.cleaned_data.get("organisation") or "No organisation"
        return f"PFD Toolkit services enquiry: {work_type} - {organisation}"

    def email_body(self) -> str:
        cleaned = self.cleaned_data
        return "\n".join(
            [
                "New PFD Toolkit services enquiry",
                "",
                f"Name: {cleaned.get('name')}",
                f"Organisation: {cleaned.get('organisation') or '-'}",
                f"Email: {cleaned.get('email')}",
                f"Work type: {cleaned.get('work_type')}",
                "",
                "Question or domain:",
                str(cleaned.get("project_summary") or "").strip(),
            ]
        )


class UserCreationForm(forms.ModelForm):
    password1 = forms.CharField(label="Password", strip=False, widget=forms.PasswordInput)
    password2 = forms.CharField(
        label="Password confirmation", strip=False, widget=forms.PasswordInput
    )

    class Meta:
        model = User
        fields = ("email", "first_name", "last_name")

    def clean_password2(self):
        password1 = self.cleaned_data.get("password1")
        password2 = self.cleaned_data.get("password2")
        if password1 and password2 and password1 != password2:
            raise forms.ValidationError("Passwords do not match.")
        return password2

    def save(self, commit=True):
        user = super().save(commit=False)
        user.set_password(self.cleaned_data["password1"])
        if commit:
            user.save()
        return user


class UserChangeForm(forms.ModelForm):
    password = ReadOnlyPasswordHashField(
        help_text=(
            "Raw passwords are not stored, so there is no way to see this user's "
            "password. You can change the password using the designated form."
        )
    )

    class Meta:
        model = User
        fields = (
            "email",
            "password",
            "first_name",
            "last_name",
            "is_active",
            "is_staff",
            "is_superuser",
            "groups",
            "user_permissions",
            "auth0_sub",
        )
