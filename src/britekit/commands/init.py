from pathlib import Path
import click
from importlib.resources import files as pkg_files
from britekit.core.util import cli_help_from_doc

INSTALL_PKG = "britekit.install"


def _iter_traversable_files(root, prefix=()):
    """Yield (relative_parts_tuple, traversable_file) for all files under root."""
    for child in root.iterdir():
        if child.is_dir():
            yield from _iter_traversable_files(child, prefix + (child.name,))
        else:
            yield (prefix + (child.name,)), child


def init_impl(dest: Path):
    """
    Setup default BriteKit directory structure and copy packaged sample files.

    This command copies files from the built-in `britekit.install` package
    (kept alongside the library code) into a folder you specify, and creates
    a default directory structure.

    Args:
        dest (Path): Directory to copy files into. Subdirectories are created as needed.

    Examples:
        britekit init --dest .
    """
    try:
        base = pkg_files(INSTALL_PKG)  # Traversable
    except ModuleNotFoundError:
        # Dev/editable install fallback: use repo-root/install
        repo_root = Path(__file__).resolve().parents[3]
        local_install = repo_root / "install"
        if local_install.exists() and local_install.is_dir():
            base = local_install
        else:
            raise click.ClickException("No packaged install found.")

    # Collect files
    files = []
    for rel_parts, trav_file in _iter_traversable_files(base):
        rel_posix = "/".join(rel_parts)  # stable across OS
        files.append((rel_posix, trav_file))

    # Copy
    dest.mkdir(parents=True, exist_ok=True)
    copied = 0
    skipped = 0
    for rel_posix, trav_file in files:
        out_path = dest / Path(rel_posix)  # preserves subdirs
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Write bytes from the packaged resource
        out_path.write_bytes(trav_file.read_bytes())
        copied += 1
        click.echo(f"copied: {out_path}")

    click.echo(f"\nDone. Copied: {copied}, Skipped: {skipped}, Dest: {dest}")


@click.command(
    name="init",
    short_help="Create default directory structure including sample files.",
    help=cli_help_from_doc(init_impl.__doc__),
)
@click.option(
    "--dest",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help="Directory to copy files into.",
)
def init_cmd(dest: Path):
    init_impl(dest)
