import pytest

from eval.adcc.data_prep.group_by_video import (
    CorpusAssertionError,
    assert_corpus_counts,
    compute_video_stem,
    group_by_video,
    usable_match_ids,
)


def test_compute_video_stem_matches_known_value():
    # Verified value from the plan (§3): md5 of this exact embed URL.
    assert compute_video_stem("https://www.youtube.com/embed/PTu0D7RmqFw") == "102e6f1680da35f52e0cf681dc38053c"


def test_group_by_video_inverts_1_to_1_into_n_to_1():
    video_links = {"m1": "url_a", "m2": "url_a", "m3": "url_b"}
    grouped = group_by_video(video_links)
    assert grouped == {"url_a": ["m1", "m2"], "url_b": ["m3"]}


def test_usable_match_ids_requires_events_link_and_disk():
    video_links = {"m1": "url_a", "m2": "url_b", "m3": "url_c"}
    events = {"m1": [{"x": 1}], "m2": []}  # m2 has empty events, m3 has none at all
    stem_a = compute_video_stem("url_a")
    stem_b = compute_video_stem("url_b")
    on_disk = {stem_a}  # only url_a's video is on disk

    usable = usable_match_ids(video_links, events, on_disk)
    assert usable == {"m1"}


def test_assert_corpus_counts_passes_on_matching_counts():
    video_links = {"m1": "url_a", "m2": "url_b"}
    events = {"m1": [{"x": 1}], "m2": [{"x": 1}]}
    on_disk = {compute_video_stem("url_a"), compute_video_stem("url_b")}

    usable, unique = assert_corpus_counts(
        video_links, events, on_disk,
        expected_usable_matches=2, expected_unique_videos=2,
    )
    assert usable == {"m1", "m2"}
    assert len(unique) == 2


def test_assert_corpus_counts_raises_on_mismatch():
    video_links = {"m1": "url_a"}
    events = {"m1": [{"x": 1}]}
    on_disk = {compute_video_stem("url_a")}

    with pytest.raises(CorpusAssertionError):
        assert_corpus_counts(video_links, events, on_disk, expected_usable_matches=999, expected_unique_videos=1)
