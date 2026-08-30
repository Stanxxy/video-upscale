from eval.adcc.data_prep.build_split import build_split


def _records(n_videos: int, matches_per_video: int = 1, events_per_match: int = 10) -> list[dict]:
    records = []
    for v in range(n_videos):
        stem = f"stem{v:03d}"
        for m in range(matches_per_video):
            records.append({
                "match_id": f"m{v:03d}_{m}",
                "video_url": f"https://example.com/{v}",
                "video_url_md5": stem,
                "name": "x vs y",
                "events": [{"t_s": i} for i in range(events_per_match)],
            })
    return records


class TestSplitByVideoNotMatch:
    def test_matches_sharing_a_video_stay_together(self):
        records = _records(n_videos=20, matches_per_video=3)
        assignment = build_split(records)
        for stem, entry in assignment.items():
            match_ids = entry["match_ids"]
            # every match_id for this stem in the source records must appear here
            expected = sorted(r["match_id"] for r in records if r["video_url_md5"] == stem)
            assert match_ids == expected

    def test_every_video_appears_in_exactly_one_split(self):
        records = _records(n_videos=30)
        assignment = build_split(records)
        splits = [entry["split"] for entry in assignment.values()]
        assert set(splits) <= {"sandbox", "dev", "test"}
        assert len(assignment) == 30

    def test_ratios_are_approximately_70_15_15(self):
        records = _records(n_videos=100)
        assignment = build_split(records)
        counts = {"sandbox": 0, "dev": 0, "test": 0}
        for entry in assignment.values():
            counts[entry["split"]] += 1
        assert counts["sandbox"] == 70
        assert counts["dev"] == 15
        assert counts["test"] == 15


class TestDeterminism:
    def test_same_seed_reproduces_byte_identical_assignment(self):
        records = _records(n_videos=50, matches_per_video=2)
        a1 = build_split(records, seed=42)
        a2 = build_split(records, seed=42)
        assert a1 == a2

    def test_different_seed_can_change_assignment(self):
        records = _records(n_videos=50, matches_per_video=2)
        a1 = build_split(records, seed=1)
        a2 = build_split(records, seed=2)
        assert a1 != a2
