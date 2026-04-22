from django import forms

from .models import MembershipAccessMode, MembershipRole, Workspace, WorkspaceLLMProvider


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


class WorkspaceCredentialUpsertForm(forms.Form):
    provider = forms.ChoiceField(
        choices=WorkspaceLLMProvider.choices,
        initial=WorkspaceLLMProvider.OPENAI,
    )
    api_key = forms.CharField(
        widget=forms.PasswordInput(render_value=False),
    )
    base_url = forms.URLField(required=False)


class WorkspaceCredentialDeleteForm(forms.Form):
    provider = forms.ChoiceField(choices=WorkspaceLLMProvider.choices)


class ActiveLLMConfigForm(forms.Form):
    provider = forms.ChoiceField(
        choices=WorkspaceLLMProvider.choices,
        initial=WorkspaceLLMProvider.OPENAI,
    )
    model_name = forms.ChoiceField(
        required=False,
        choices=(
            ("gpt-4.1-mini", "gpt-4.1-mini (recommended - lower cost)"),
            ("gpt-4.1", "gpt-4.1 (higher accuracy)"),
        ),
        initial="gpt-4.1-mini",
    )
    max_parallel_workers = forms.IntegerField(
        required=False,
        min_value=1,
        max_value=32,
        initial=1,
    )
    api_key = forms.CharField(
        required=False,
        widget=forms.PasswordInput(render_value=False),
    )
    base_url = forms.URLField(required=False)
    next_url = forms.CharField(required=False)


class ActiveLLMCredentialDeleteForm(forms.Form):
    provider = forms.ChoiceField(
        choices=WorkspaceLLMProvider.choices,
        initial=WorkspaceLLMProvider.OPENAI,
    )
    next_url = forms.CharField(required=False)


class WorkspaceReportExclusionCreateForm(forms.Form):
    report_identity = forms.CharField(max_length=512)
    reason = forms.CharField(required=False)
    report_title = forms.CharField(required=False)
    report_date = forms.CharField(required=False, max_length=32)
    report_url = forms.URLField(required=False)
    next_url = forms.CharField(required=False)
