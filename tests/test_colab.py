import sys
import types

from sitex.core import colab


def test_enable_widgets_is_a_noop_outside_colab(monkeypatch):
    monkeypatch.delitem(sys.modules, "google.colab", raising=False)
    assert colab.enable_widgets() is False


def test_enable_widgets_turns_on_custom_manager_in_colab(monkeypatch):
    calls = []
    output = types.SimpleNamespace(enable_custom_widget_manager=lambda: calls.append(1))
    fake_colab = types.ModuleType("google.colab")
    fake_colab.output = output
    fake_google = types.ModuleType("google")
    fake_google.colab = fake_colab
    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.colab", fake_colab)

    assert colab.enable_widgets() is True
    assert calls == [1]
