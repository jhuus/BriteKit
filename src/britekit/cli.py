import click

try:
    from .__about__ import __version__  # type: ignore
except Exception:
    try:
        from importlib.metadata import version as _pkg_version  # type: ignore

        __version__ = _pkg_version("britekit")  # type: ignore[assignment]
    except Exception:
        __version__ = "0.0.0"  # last-resort fallback

from .commands.analyze_cmd import _analyze_cmd
from .commands.audioset_cmd import _audioset_cmd
from .commands.calibrate_cmd import _calibrate_cmd
from .commands.ckpt_ops_cmd import _ckpt_avg_cmd, _ckpt_freeze_cmd, _ckpt_onnx_cmd
from .commands.db_add_cmd import (
    _add_cat_cmd,
    _add_class_cmd,
    _add_src_cmd,
    _add_stype_cmd,
)
from .commands.db_delete_cmd import (
    _del_cat_cmd,
    _del_class_cmd,
    _del_rec_cmd,
    _del_seg_cmd,
    _del_sgroup_cmd,
    _del_src_cmd,
    _del_stype_cmd,
)
from .commands.embed_cmd import _embed_cmd
from .commands.extract_cmd import _extract_all_cmd, _extract_by_image_cmd
from .commands.find_dup_cmd import _find_dup_cmd
from .commands.inat_cmd import _inat_cmd
from .commands.init_cmd import _init_cmd
from .commands.pickle_cmd import _pickle_cmd
from .commands.plot_cmd import _plot_db_cmd, _plot_dir_cmd, _plot_rec_cmd
from .commands.reextract_cmd import _reextract_cmd
from .commands.report_cmd import (
    _rpt_ann_cmd,
    _rpt_db_cmd,
    _rpt_epochs_cmd,
    _rpt_labels_cmd,
    _rpt_test_cmd,
)
from .commands.search_cmd import _search_cmd
from .commands.train_cmd import _find_lr_cmd, _train_cmd
from .commands.tune_cmd import _tune_cmd
from .commands.wav2mp3_cmd import _wav2mp3_cmd
from .commands.xeno_cmd import _xeno_cmd
from .commands.youtube_cmd import _youtube_cmd


@click.group()
@click.version_option(__version__)  # enabled the "britekit --version" command
def cli():
    """BriteKit CLI tools."""
    pass


cli.add_command(_add_cat_cmd)
cli.add_command(_add_stype_cmd)
cli.add_command(_add_src_cmd)
cli.add_command(_add_class_cmd)
cli.add_command(_analyze_cmd)
cli.add_command(_audioset_cmd)

cli.add_command(_calibrate_cmd)
cli.add_command(_ckpt_avg_cmd)
cli.add_command(_ckpt_freeze_cmd)
cli.add_command(_ckpt_onnx_cmd)

cli.add_command(_del_cat_cmd)
cli.add_command(_del_class_cmd)
cli.add_command(_del_rec_cmd)
cli.add_command(_del_seg_cmd)
cli.add_command(_del_sgroup_cmd)
cli.add_command(_del_src_cmd)
cli.add_command(_del_stype_cmd)

cli.add_command(_embed_cmd)
cli.add_command(_extract_all_cmd)
cli.add_command(_extract_by_image_cmd)

cli.add_command(_find_dup_cmd)
cli.add_command(_find_lr_cmd)

cli.add_command(_inat_cmd)
cli.add_command(_init_cmd)

cli.add_command(_pickle_cmd)
cli.add_command(_plot_dir_cmd)
cli.add_command(_plot_db_cmd)
cli.add_command(_plot_rec_cmd)

cli.add_command(_search_cmd)

cli.add_command(_reextract_cmd)
cli.add_command(_rpt_ann_cmd)
cli.add_command(_rpt_db_cmd)
cli.add_command(_rpt_epochs_cmd)
cli.add_command(_rpt_labels_cmd)
cli.add_command(_rpt_test_cmd)

cli.add_command(_train_cmd)
cli.add_command(_tune_cmd)

cli.add_command(_wav2mp3_cmd)

cli.add_command(_xeno_cmd)

cli.add_command(_youtube_cmd)
