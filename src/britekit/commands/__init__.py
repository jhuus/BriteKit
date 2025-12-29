#!/usr/bin/env python3

from ._analyze import analyze
from ._analyze_db import analyze_db
from ._audioset import audioset
from ._calibrate import calibrate
from ._ckpt_ops import ckpt_avg, ckpt_freeze, ckpt_onnx
from ._db_add import add_cat, add_class, add_src, add_stype
from ._db_delete import (
    del_cat,
    del_class,
    del_rec,
    del_sgroup,
    del_seg,
    del_src,
    del_stype,
)
from ._dedup import dedup_rec, dedup_seg
from ._embed import embed
from ._ensemble import ensemble
from ._extract import extract_all, extract_by_csv, extract_by_image

from ._inat import inat
from ._init import init
from ._pickle import pickle_occurrence, pickle_train
from ._plot import plot_db, plot_dir, plot_rec, plot_test
from ._reextract import reextract
from ._reports import rpt_ann, rpt_db, rpt_epochs, rpt_labels, rpt_test
from ._search import search
from ._train import train, find_lr
from ._tune import tune
from ._wav2mp3 import wav2mp3
from ._xeno import xeno
from ._youtube import youtube

__all__ = [
    "add_cat",
    "add_class",
    "add_src",
    "add_stype",
    "analyze",
    "analyze_db",
    "audioset",
    "calibrate",
    "ckpt_avg",
    "ckpt_freeze",
    "ckpt_onnx",
    "copy_samples",
    "dedup_rec",
    "dedup_seg",
    "del_cat",
    "del_class",
    "del_rec",
    "del_seg",
    "del_sgroup",
    "del_src",
    "del_stype",
    "embed",
    "ensemble",
    "extract_all",
    "extract_by_csv",
    "extract_by_image",
    "find_lr",
    "inat",
    "init",
    "pickle_occurrence",
    "pickle_train",
    "plot_db",
    "plot_dir",
    "plot_rec",
    "plot_test",
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
