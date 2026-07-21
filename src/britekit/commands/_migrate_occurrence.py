"""CLI and Python API for occurrence database migration."""

from pathlib import Path

import click

from britekit.occurrence_db.occurrence_migration import (
    OccurrenceMigrationReport,
    migrate_occurrence_v1_to_v2,
)


def migrate_occurrence(
    input_path: str,
    output_path: str,
    metadata_path: str,
) -> OccurrenceMigrationReport:
    """Migrate an occurrence database from schema v1 to schema v2.

    Args:
    - input_path: Existing schema-v1 occurrence database.
    - output_path: New schema-v2 database to create.
    - metadata_path: JSON file defining parent areas and level labels.
    """
    return migrate_occurrence_v1_to_v2(input_path, output_path, metadata_path)


@click.command(
    name="migrate-occurrence",
    short_help="Migrate an occurrence database to the hierarchical schema.",
)
@click.option(
    "--input",
    "input_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Schema-v1 occurrence database.",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
    help="Schema-v2 occurrence database to create.",
)
@click.option(
    "--metadata",
    "metadata_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="JSON file defining region-pack and administrative-area metadata.",
)
def _migrate_occurrence_cmd(
    input_path: Path,
    output_path: Path,
    metadata_path: Path,
) -> None:
    report = migrate_occurrence(str(input_path), str(output_path), str(metadata_path))
    click.echo(f"Source occurrence areas: {report.source_area_count}")
    click.echo(f"Parent areas added: {report.parent_area_count}")
    click.echo(f"Total administrative areas: {report.total_area_count}")
    click.echo(f"Classes copied: {report.class_count}")
    click.echo(f"Occurrence rows copied: {report.occurrence_count}")
    click.echo(f"Created: {output_path}")
