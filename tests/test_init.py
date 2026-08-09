from britekit.commands import init


def test_init_allows_existing_directories(tmp_path):
    # Exercise an existing destination and non-empty directories from the
    # structure that init creates.
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    existing_file = data_dir / "existing.txt"
    existing_file.write_text("keep me")

    samples_dir = tmp_path / "yaml" / "samples"
    samples_dir.mkdir(parents=True)

    init(tmp_path)
    init(tmp_path)

    assert (tmp_path / "yaml" / "base_config.yaml").is_file()
    assert (samples_dir / "cfg_infer.yaml").is_file()
    assert existing_file.read_text() == "keep me"
