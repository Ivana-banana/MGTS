import pytest

from mgts_nav.safe_defaults import resolve_confirm_start, resolve_recover


@pytest.mark.parametrize('value, expected', [
    (None, True),
    (True, True),
    (False, False),
])
def test_recover_defaults_on(value, expected):
    assert resolve_recover(value) is expected


def test_confirm_defaults_on_for_automatic_start():
    assert resolve_confirm_start(None) is True
    assert resolve_confirm_start(True) is True
    assert resolve_confirm_start(False) is False


def test_confirm_off_with_provided_start_or_search():
    assert resolve_confirm_start(None, initial=[0, 0, 0]) is False
    assert resolve_confirm_start(False, initial=[0, 0, 0]) is False
    assert resolve_confirm_start(None, start_search='delayed') is False
    with pytest.raises(ValueError, match='automatic initialization'):
        resolve_confirm_start(True, initial=[0, 0, 0])
    with pytest.raises(ValueError, match='automatic initialization'):
        resolve_confirm_start(True, start_search='coarse')
