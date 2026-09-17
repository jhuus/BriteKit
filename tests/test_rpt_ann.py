import csv

from click.testing import CliRunner

from britekit.commands._reports import _rpt_ann_cmd


def test_summary_counts_distinct_recordings_per_class(tmp_path):
    annotations = tmp_path / "annotations.csv"
    annotations.write_text(
        "recording,class,start_time,end_time\n"
        "001,BCCH,0.0,3.0\n"
        "001,BCCH,3,6\n"
        "002,BCCH,0,2\n"
        "001,BRCR,6,9\n"
        "empty,,0,3\n"
    )
    output = tmp_path / "report"
    result = CliRunner().invoke(
        _rpt_ann_cmd, ["-a", str(annotations), "-o", str(output)]
    )
    assert result.exit_code == 0, result.output
    with (output / "test_summary.csv").open() as file:
        reader = csv.DictReader(file)
        assert reader.fieldnames == ["class", "seconds", "recordings"]
        assert list(reader) == [
            {"class": "BCCH", "seconds": "8.0", "recordings": "2"},
            {"class": "BRCR", "seconds": "3.0", "recordings": "1"},
        ]
    with (output / "test_details.csv").open() as file:
        rows = list(csv.DictReader(file))
    assert rows == [
        {"recording": "001", "class": "BCCH", "seconds": "6.0"},
        {"recording": "001", "class": "BRCR", "seconds": "3.0"},
        {"recording": "002", "class": "BCCH", "seconds": "2.0"},
        {"recording": "empty", "class": "", "seconds": "3.0"},
    ]


def test_empty_annotations_still_write_summary_columns(tmp_path):
    annotations = tmp_path / "annotations.csv"
    annotations.write_text("recording,class,start_time,end_time\n")
    output = tmp_path / "report"
    result = CliRunner().invoke(
        _rpt_ann_cmd, ["-a", str(annotations), "-o", str(output)]
    )
    assert result.exit_code == 0, result.output
    assert (output / "test_summary.csv").read_text() == "class,seconds,recordings\n"
