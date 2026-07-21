"""CLI and Python API for compiling a lightweight location catalog."""

from pathlib import Path

import click

from britekit.occurrence_db.location_catalog import (
    LocationCatalogReport,
    compile_location_catalog as _compile_location_catalog,
)


def compile_location_catalog(
    input_path: str, output_path: str
) -> LocationCatalogReport:
    """Compile GUI location metadata from a schema-v2 occurrence database."""
    return _compile_location_catalog(input_path, output_path)


@click.command(
    name="compile-location-catalog",
    short_help="Compile a lightweight administrative-area catalog.",
)
@click.option(
    "--input",
    "input_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Schema-v2 occurrence database.",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
    help="Location catalog database to create.",
)
def _compile_location_catalog_cmd(input_path: Path, output_path: Path) -> None:
    report = compile_location_catalog(str(input_path), str(output_path))
    click.echo(f"Region: {report.region_code}")
    click.echo(f"Administrative areas: {report.area_count}")
    click.echo(f"Selectable areas: {report.selectable_area_count}")
    click.echo(f"Administrative level labels: {report.level_count}")
    click.echo(f"Created: {output_path}")
