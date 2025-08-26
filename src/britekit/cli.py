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


cli.add_command(commands.add_cat)
cli.add_command(commands.add_stype)
cli.add_command(commands.add_src)
cli.add_command(commands.add_class)
cli.add_command(commands.analyze)
cli.add_command(commands.audioset)

cli.add_command(commands.ckpt_avg)
cli.add_command(commands.ckpt_freeze)
cli.add_command(commands.ckpt_onnx)

cli.add_command(commands.del_cat)
cli.add_command(commands.del_class)
cli.add_command(commands.del_rec)
cli.add_command(commands.del_seg)
cli.add_command(commands.del_sgroup)
cli.add_command(commands.del_src)
cli.add_command(commands.del_stype)

cli.add_command(commands.embed)
cli.add_command(commands.extract_all)
cli.add_command(commands.extract_by_image)

cli.add_command(commands.find_dup)
cli.add_command(commands.find_lr)

cli.add_command(commands.inat)
cli.add_command(commands.init)

cli.add_command(commands.pickle)
cli.add_command(commands.plot_dir)
cli.add_command(commands.plot_db)
cli.add_command(commands.plot_rec)

cli.add_command(commands.search)

cli.add_command(commands.reextract)
cli.add_command(commands.rpt_ann)
cli.add_command(commands.rpt_cal)
cli.add_command(commands.rpt_db)
cli.add_command(commands.rpt_labels)
cli.add_command(commands.rpt_test)

cli.add_command(commands.train)
cli.add_command(commands.tune)

cli.add_command(commands.wav2mp3)

cli.add_command(commands.xeno)
