from molecule_gen.data.download import format_bytes


def test_format_bytes():
    assert format_bytes(512) == "512 B"
    assert format_bytes(2048) == "2.0 KiB"
    assert format_bytes(2 * 1024 * 1024) == "2.0 MiB"
    assert format_bytes(3 * 1024 * 1024 * 1024) == "3.00 GiB"
