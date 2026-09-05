from types import SimpleNamespace

import pytest

import dflash2


class FakeCache:
    def __init__(self, current_length: int) -> None:
        self.current_length = current_length
        self.crop_calls: list[int] = []

    def get_seq_length(self) -> int:
        return self.current_length

    def crop(self, tokens_to_remove: int) -> None:
        self.crop_calls.append(tokens_to_remove)


def test_create_generation_cache_uses_config_and_records_past(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created = SimpleNamespace(config=None, recording=False)

    class StubDynamicCache:
        def __init__(self, *, config) -> None:
            created.config = config

        def activate_past_recording(self) -> None:
            created.recording = True

    monkeypatch.setattr(dflash2, "DynamicCache", StubDynamicCache)
    config = object()

    cache = dflash2.create_generation_cache(config)

    assert isinstance(cache, StubDynamicCache)
    assert created.config is config
    assert created.recording


@pytest.mark.parametrize(
    ("current_length", "retained_length", "expected_crop"),
    [(12, 9, -3), (12, 12, 0)],
)
def test_retain_cache_prefix_uses_negative_removal_count(
    current_length: int,
    retained_length: int,
    expected_crop: int,
) -> None:
    cache = FakeCache(current_length)

    dflash2.retain_cache_prefix(cache, retained_length)

    assert cache.crop_calls == [expected_crop]


@pytest.mark.parametrize("retained_length", [-1, 13])
def test_retain_cache_prefix_rejects_invalid_length(
    retained_length: int,
) -> None:
    cache = FakeCache(12)

    with pytest.raises(ValueError, match="retained cache length"):
        dflash2.retain_cache_prefix(cache, retained_length)
