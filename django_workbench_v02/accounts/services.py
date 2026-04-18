from __future__ import annotations

from dataclasses import dataclass

from django.conf import settings
from django.contrib.auth import get_user_model


User = get_user_model()


@dataclass(frozen=True)
class Auth0Profile:
    sub: str | None
    email: str
    given_name: str
    family_name: str
    name: str


def split_name(name: str) -> tuple[str, str]:
    cleaned = (name or "").strip()
    if not cleaned:
        return "", ""
    parts = cleaned.split(maxsplit=1)
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], parts[1]


def normalize_auth0_profile(payload: dict) -> Auth0Profile:
    email = (payload.get("email") or "").strip().lower()
    given_name = (payload.get("given_name") or "").strip()
    family_name = (payload.get("family_name") or "").strip()
    full_name = (payload.get("name") or "").strip()

    if not given_name and not family_name and full_name:
        given_name, family_name = split_name(full_name)

    return Auth0Profile(
        sub=(payload.get("sub") or "").strip() or None,
        email=email,
        given_name=given_name,
        family_name=family_name,
        name=full_name,
    )


def sync_user_from_auth0(profile: Auth0Profile):
    lookup_candidates = []
    if profile.sub:
        lookup_candidates.append(("auth0_sub", profile.sub))
    if profile.email:
        lookup_candidates.append(("email", profile.email))

    user = None
    for field_name, value in lookup_candidates:
        try:
            user = User.objects.get(**{field_name: value})
            break
        except User.DoesNotExist:
            continue

    if user is None:
        user = User(
            email=profile.email,
            first_name=profile.given_name,
            last_name=profile.family_name,
            auth0_sub=profile.sub,
            is_active=True,
        )
        user.set_unusable_password()
        user.save()
    else:
        changed_fields = []
        if profile.sub and user.auth0_sub != profile.sub:
            user.auth0_sub = profile.sub
            changed_fields.append("auth0_sub")
        if profile.given_name and user.first_name != profile.given_name:
            user.first_name = profile.given_name
            changed_fields.append("first_name")
        if profile.family_name and user.last_name != profile.family_name:
            user.last_name = profile.family_name
            changed_fields.append("last_name")
        if changed_fields:
            user.save(update_fields=changed_fields)

    if user.email.lower() == settings.PFD_ADMIN_EMAIL:
        if not user.is_staff or not user.is_superuser:
            user.is_staff = True
            user.is_superuser = True
            user.save(update_fields=["is_staff", "is_superuser"])
    elif user.is_superuser:
        # Ensure only the configured admin email is superuser by default.
        user.is_superuser = False
        user.is_staff = False
        user.save(update_fields=["is_staff", "is_superuser"])

    return user
