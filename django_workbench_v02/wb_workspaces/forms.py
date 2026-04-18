from django import forms

from .models import MembershipAccessMode, MembershipRole, Workspace


class WorkspaceCreateForm(forms.ModelForm):
    class Meta:
        model = Workspace
        fields = ["title", "slug", "description", "visibility", "is_listed"]


class WorkspaceMemberAddForm(forms.Form):
    email = forms.EmailField()
    role = forms.ChoiceField(choices=MembershipRole.choices)
    access_mode = forms.ChoiceField(choices=MembershipAccessMode.choices)
    can_manage_members = forms.BooleanField(required=False)
    can_manage_shares = forms.BooleanField(required=False)
    can_run_workflows = forms.BooleanField(required=False, initial=True)


class WorkspaceMemberUpdateForm(forms.Form):
    role = forms.ChoiceField(choices=MembershipRole.choices)
    access_mode = forms.ChoiceField(choices=MembershipAccessMode.choices)
    can_manage_members = forms.BooleanField(required=False)
    can_manage_shares = forms.BooleanField(required=False)
    can_run_workflows = forms.BooleanField(required=False, initial=True)
