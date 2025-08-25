import click
try:
    from .__about__ import __version__  # type: ignore
except Exception:
    try:
        from importlib.metadata import version as _pkg_version  # type: ignore
        __version__ = _pkg_version("britekit")  # type: ignore[assignment]
    except Exception:
        __version__ = "0.0.0"  # last-resort fallback
from britekit import commands


@click.group()
@click.version_option(__version__)  # enabled the "britekit --version" command
def cli():
    """BriteKit CLI tools."""
    pass


cli.add_command(commands.add_cat_cmd)
cli.add_command(commands.add_stype_cmd)
cli.add_command(commands.add_src_cmd)
cli.add_command(commands.add_class_cmd)
cli.add_command(commands.analyze_cmd)
cli.add_command(commands.audioset_cmd)

cli.add_command(commands.ckpt_avg_cmd)
cli.add_command(commands.ckpt_freeze_cmd)
cli.add_command(commands.ckpt_onnx_cmd)

cli.add_command(commands.del_cat_cmd)
cli.add_command(commands.del_class_cmd)
cli.add_command(commands.del_rec_cmd)
cli.add_command(commands.del_sgroup_cmd)
cli.add_command(commands.del_stype_cmd)
cli.add_command(commands.del_src_cmd)
cli.add_command(commands.del_spec_cmd)

cli.add_command(commands.embed_cmd)
cli.add_command(commands.extract_all_cmd)
cli.add_command(commands.extract_by_image_cmd)

cli.add_command(commands.find_dup_cmd)
cli.add_command(commands.find_lr_cmd)

cli.add_command(commands.inat_cmd)
cli.add_command(commands.init_cmd)

cli.add_command(commands.pickle_cmd)
cli.add_command(commands.plot_dir_cmd)
cli.add_command(commands.plot_db_cmd)
cli.add_command(commands.plot_rec_cmd)

cli.add_command(commands.search_cmd)

cli.add_command(commands.reextract_cmd)
cli.add_command(commands.rpt_ann_cmd)
cli.add_command(commands.rpt_cal_cmd)
cli.add_command(commands.rpt_db_cmd)
cli.add_command(commands.rpt_labels_cmd)
cli.add_command(commands.rpt_test_cmd)

cli.add_command(commands.train_cmd)
cli.add_command(commands.tune_cmd)

cli.add_command(commands.wav2mp3_cmd)

cli.add_command(commands.xeno_cmd)
