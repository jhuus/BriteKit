from .analyze import analyze_cmd
from .audioset import audioset_cmd
from .calibrate import calibrate_cmd
from .ckpt_ops import ckpt_avg_cmd, ckpt_freeze_cmd, ckpt_onnx_cmd
from .db_add import add_cat_cmd, add_class_cmd, add_src_cmd, add_stype_cmd
from .db_delete import (
    del_cat_cmd,
    del_class_cmd,
    del_rec_cmd,
    del_sgroup_cmd,
    del_seg_cmd,
    del_src_cmd,
    del_stype_cmd,
)
from .embed import embed_cmd
from .extract import extract_all_cmd, extract_by_image_cmd
from .find_dup import find_dup_cmd
from .inat import inat_cmd
from .init import init_cmd
from .pickle import pickle_cmd
from .plot import plot_db_cmd, plot_dir_cmd, plot_rec_cmd
from .reextract import reextract_cmd
from .reports import (
    rpt_ann_cmd,
    rpt_db_cmd,
    rpt_epochs_cmd,
    rpt_labels_cmd,
    rpt_test_cmd
)
from .search import search_cmd
from .train import train_cmd, find_lr_cmd
from .tune import tune_cmd
from .wav2mp3 import wav2mp3_cmd
from .xeno import xeno_cmd
from .youtube import youtube_cmd

# We want to expose them to the API as britekit.commands.analyze etc.
# But MyPy gets confused when the function name matches the module name,
# so we alias the commands here.
add_cat = add_cat_cmd
add_class = add_class_cmd
add_src = add_src_cmd
add_stype = add_stype_cmd
analyze = analyze_cmd
audioset = audioset_cmd
calibrate = calibrate_cmd
ckpt_avg = ckpt_avg_cmd
ckpt_freeze = ckpt_freeze_cmd
ckpt_onnx = ckpt_onnx_cmd
del_cat = del_cat_cmd
del_class = del_class_cmd
del_rec = del_rec_cmd
del_seg = del_seg_cmd
del_sgroup = del_sgroup_cmd
del_src = del_src_cmd
del_stype = del_stype_cmd
embed = embed_cmd
extract_all = extract_all_cmd
extract_by_image = extract_by_image_cmd
find_dup = find_dup_cmd
find_lr = find_lr_cmd
inat = inat_cmd
init = init_cmd
pickle = pickle_cmd
plot_db = plot_db_cmd
plot_dir = plot_dir_cmd
plot_rec = plot_rec_cmd
reextract = reextract_cmd
rpt_ann = rpt_ann_cmd
rpt_db = rpt_db_cmd
rpt_epochs = rpt_epochs_cmd
rpt_labels = rpt_labels_cmd
rpt_test = rpt_test_cmd
search = search_cmd
train = train_cmd
tune = tune_cmd
wav2mp3 = wav2mp3_cmd
xeno = xeno_cmd
youtube = youtube_cmd

__all__ = [
    "add_cat",
    "add_class",
    "add_src",
    "add_stype",
    "analyze",
    "audioset",
    "calibrate",
    "ckpt_avg",
    "ckpt_freeze",
    "ckpt_onnx",
    "copy_samples",
    "del_cat",
    "del_class",
    "del_rec",
    "del_seg",
    "del_sgroup",
    "del_src",
    "del_stype",
    "embed",
    "extract_all",
    "extract_by_image",
    "find_dup",
    "inat",
    "pickle",
    "plot_db",
    "plot_dir",
    "plot_file",
    "reextract",
    "rpt_ann",
    "rpt_cal",
    "rpt_db",
    "rpt_epochs",
    "rpt_labels",
    "rpt_test",
    "search",
    "train",
    "tune",
    "wav2mp3",
    "xeno",
    "youtube",
]
