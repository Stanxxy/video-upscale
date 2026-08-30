from eval.adcc.data_prep.apply_gracie_mapping import load_mapping
from eval.adcc.data_prep.build_gold_snapshot import build_gold_snapshot
from eval.adcc.data_prep.group_by_video import compute_video_stem

MAPPING = load_mapping()


def test_build_gold_snapshot_end_to_end_on_a_tiny_synthetic_corpus():
    url = "https://www.youtube.com/embed/FAKE12345"
    stem = compute_video_stem(url)

    metadata_and_events = {
        "metadata": {"m1": {"name": "Fake A vs Fake B"}},
        "events": {"m1": [
            {"timestamp": "01:00:00", "sequenceType": "TAKEDOWN", "startingPosition": "STANDING",
             "movements": "SINGLE LEG", "endingPosition": "STANDING"},
            {"timestamp": "01:00", "sequenceType": "POSITION ENTRY", "startingPosition": "TURTLE",
             "movements": "BERIMBOLO", "endingPosition": "BACK"},
            {"timestamp": "01:30", "sequenceType": "SUBMISSION", "startingPosition": "BACK",
             "movements": "N/A", "endingPosition": "RNC"},
        ]},
    }
    video_links = {"m1": url}
    on_disk_stems = {stem}

    records = build_gold_snapshot(
        metadata_and_events, video_links, on_disk_stems, MAPPING,
        expected_usable_matches=1, expected_unique_videos=1,
    )

    assert len(records) == 1
    rec = records[0]
    assert rec["match_id"] == "m1"
    assert rec["video_url_md5"] == stem
    assert rec["name"] == "Fake A vs Fake B"

    # chronological order: HH:MM:SS "01:00:00" (3600s) sorts AFTER MM:SS "01:00" (60s) and "01:30" (90s)
    events = rec["events"]
    assert [e["t_s"] for e in events] == sorted(e["t_s"] for e in events)
    action_classes = [e["action_class"] for e in events]
    assert "back_take" in action_classes
    assert "submission_choke" in action_classes
    assert "takedown_attempt" in action_classes
