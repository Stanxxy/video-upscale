"""Idle DAG poll pages stay out of INFO: on gx10 they were ~300 lines/min from the trial host."""
import logging

from service.dag_server import _poll_level


def test_idle_poll_page_is_debug():
    assert _poll_level(()) == logging.DEBUG
    assert _poll_level((), None) == logging.DEBUG


def test_poll_page_with_work_is_info():
    assert _poll_level(("attempt",)) == logging.INFO


def test_poll_page_error_is_warning_even_when_idle():
    assert _poll_level((), "store_unavailable") == logging.WARNING
    assert _poll_level(("attempt",), "store_unavailable") == logging.WARNING
