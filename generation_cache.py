from transformers import DynamicCache


def create_generation_cache(config) -> DynamicCache:
    cache = DynamicCache(config=config)
    cache.activate_past_recording()
    return cache


def retain_cache_prefix(cache: DynamicCache, retained_length: int) -> None:
    current_length = cache.get_seq_length()
    if retained_length < 0 or retained_length > current_length:
        raise ValueError(
            "retained cache length must be between zero and the current "
            f"length ({current_length}), got {retained_length}"
        )
    cache.crop(-(current_length - retained_length))
