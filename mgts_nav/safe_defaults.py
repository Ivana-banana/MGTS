"""Resolve localization safety defaults for CLI and solve."""


def resolve_confirm_start(confirm_start, *, initial=None, start_search=None):
    """Confirm automatic single-scan starts unless explicitly disabled.

    Provided starts and ``--start-search`` never confirm. Passing
    ``confirm_start=True`` together with either is an error.
    """
    locked = initial is not None or start_search is not None
    if locked:
        if confirm_start is True:
            raise ValueError(
                '--confirm-start requires automatic initialization. '
                'Omit --initial and --start-search.')
        return False
    if confirm_start is None:
        return True
    return bool(confirm_start)


def resolve_recover(recover):
    """Enable global recovery after localization loss unless disabled."""
    if recover is None:
        return True
    return bool(recover)
