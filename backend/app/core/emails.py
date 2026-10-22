"""User email normalization (docs/DATABASE.md, service rule 1)."""


def normalize_user_email(email: str) -> str:
    """Canonical form of users.email: trimmed and lowercased.

    Used both when storing a user and when looking one up at login, so the two can never
    disagree. Contact and lead emails are not normalized (stored as entered).
    """
    return email.strip().lower()
