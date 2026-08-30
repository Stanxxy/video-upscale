import pytest

from eval.adcc.data_prep.parse_timestamp import TimestampParseError, parse_timestamp


class TestMMSS:
    def test_basic(self):
        assert parse_timestamp("02:22") == 2 * 60 + 22

    def test_zero(self):
        assert parse_timestamp("00:00") == 0

    def test_max_valid(self):
        assert parse_timestamp("59:59") == 59 * 60 + 59


class TestHHMMSS:
    def test_basic(self):
        assert parse_timestamp("08:01:12") == 8 * 3600 + 1 * 60 + 12

    def test_multi_hour_stream(self):
        # real corpus value shape: an 8-hour livestream timestamp
        assert parse_timestamp("06:12:09") == 6 * 3600 + 12 * 60 + 9

    def test_zero_hours(self):
        assert parse_timestamp("00:23:17") == 23 * 60 + 17


class TestMalformedRaises:
    @pytest.mark.parametrize("bad", [
        "125",                # 1 field
        "1:2:3:4",             # 4 fields
        "",                    # 0 fields (splits to [''])
        "1:2.5",               # non-integer field
        "aa:bb",               # non-numeric
        "1:-5",                # negative not accepted by isdigit()
        "60:00",               # minutes out of range
        "10:60",               # seconds out of range
        "1:2:3:",              # trailing colon -> 4 fields incl empty
    ])
    def test_raises(self, bad):
        with pytest.raises(TimestampParseError):
            parse_timestamp(bad)

    def test_non_string_raises(self):
        with pytest.raises(TimestampParseError):
            parse_timestamp(142)  # type: ignore[arg-type]

    def test_none_raises(self):
        with pytest.raises(TimestampParseError):
            parse_timestamp(None)  # type: ignore[arg-type]
