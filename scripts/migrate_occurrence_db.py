# Quick script to migrate a training database

from britekit.core.config_loader import get_config
from britekit.occurrence_db.occurrence_db import OccurrenceDatabase

cfg, fn_cfg = get_config()
fn_cfg.echo = print

with OccurrenceDatabase("data/occurrence.db") as db:
    classes = db.get_all_classes()
    print(f"Number of classes = {len(classes)}")
