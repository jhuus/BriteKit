# Quick script to migrate a training database

from britekit.core.config_loader import get_config
from britekit.training_db.training_db import TrainingDatabase

cfg, fn_cfg = get_config()
fn_cfg.echo = print

db_path = "data/training.db"
with TrainingDatabase(db_path) as db:
    count = db.get_class_count()
    print(f"Number of classes = {count}")
