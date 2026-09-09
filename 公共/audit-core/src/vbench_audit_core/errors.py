class AuditError(Exception):
    """Expected user-facing evaluation error."""


class InputError(AuditError):
    pass


class DependencyError(AuditError):
    pass
