"""
Invoke - Tasks
==============
"""

from __future__ import annotations

import contextlib
import fnmatch
import inspect
import os
import re
import shutil
import sys
import tarfile
import tomllib
import typing
import uuid
from pathlib import Path
from shlex import quote

import biblib.bib

import colour
from colour.utilities import message_box

if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec  # pyright: ignore

if typing.TYPE_CHECKING:
    from invoke.context import Context

from invoke.tasks import task

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "APPLICATION_NAME",
    "APPLICATION_VERSION",
    "PYTHON_PACKAGE_NAME",
    "PYPI_PACKAGE_NAME",
    "PYPI_ARCHIVE_NAME",
    "BIBLIOGRAPHY_NAME",
    "literalise",
    "clean",
    "formatting",
    "quality",
    "api",
    "prek",
    "tests",
    "examples",
    "preflight",
    "docs",
    "todo",
    "requirements",
    "build",
    "virtualise",
    "tag",
    "pre_release",
    "release",
    "sha256",
]

APPLICATION_NAME: str = colour.__application_name__

APPLICATION_VERSION: str = colour.__version__

PYTHON_PACKAGE_NAME: str = colour.__name__

PYPI_PACKAGE_NAME: str = "colour-science"
PYPI_ARCHIVE_NAME: str = PYPI_PACKAGE_NAME.replace("-", "_")

BIBLIOGRAPHY_NAME: str = "BIBLIOGRAPHY.bib"


@task
def literalise(ctx: Context) -> None:
    """
    Write various literals in the `colour.hints` module.

    Parameters
    ----------
    ctx
        Context.
    """

    message_box("Literalising...")
    with ctx.cd("utilities"):
        ctx.run("./literalise.py")

    ctx.run("prek run --files colour/hints/__init__.py", warn=True)


@task
def clean(
    ctx: Context,
    docs: bool = True,
    bytecode: bool = False,
    pytest: bool = True,
) -> None:
    """
    Clean the project.

    Parameters
    ----------
    ctx
        Context.
    docs
        Whether to clean the *docs* directory.
    bytecode
        Whether to clean the bytecode files, e.g., *.pyc* files.
    pytest
        Whether to clean the *Pytest* cache directory.
    """

    message_box("Cleaning project...")

    patterns = ["build", "*.egg-info", "dist"]

    if docs:
        patterns.append("docs/_build")
        patterns.append("docs/generated")

    if bytecode:
        patterns.append("**/__pycache__")
        patterns.append("**/*.pyc")

    if pytest:
        patterns.append(".pytest_cache")

    for pattern in patterns:
        ctx.run(f"rm -rf {pattern}")


@task
def formatting(
    ctx: Context,
    asciify: bool = True,
    bibtex: bool = True,
) -> None:
    """
    Convert unicode characters to ASCII and cleanup the *BibTeX* file.

    Parameters
    ----------
    ctx
        Context.
    asciify
        Whether to convert unicode characters to ASCII.
    bibtex
        Whether to cleanup the *BibTeX* file.
    """

    if asciify:
        message_box("Converting unicode characters to ASCII...")
        with ctx.cd("utilities"):
            ctx.run("./unicode_to_ascii.py")

    if bibtex:
        message_box('Cleaning up "BibTeX" file...')
        bibtex_path = BIBLIOGRAPHY_NAME
        with open(bibtex_path) as bibtex_file:
            entries = biblib.bib.Parser().parse(bibtex_file.read()).get_entries()

        for entry in sorted(entries.values(), key=lambda x: x.key):
            with contextlib.suppress(KeyError):
                del entry["file"]

            for key, value in entry.items():
                entry[key] = re.sub("(?<!\\\\)\\&", "\\&", value)

        with open(bibtex_path, "w") as bibtex_file:
            for entry in sorted(entries.values(), key=lambda x: x.key):
                bibtex_file.write(entry.to_bib())
                bibtex_file.write("\n")


@task
def quality(
    ctx: Context,
    pyright: bool = True,
    rstlint: bool = True,
) -> None:
    """
    Check the codebase with *Pyright* and lints various *restructuredText*
    files with *rst-lint*.

    Parameters
    ----------
    ctx
        Context.
    pyright
        Whether to check the codebase with *Pyright*.
    rstlint
        Whether to lint various *restructuredText* files with *rst-lint*.
    """

    if pyright:
        message_box('Checking codebase with "Pyright"...')
        ctx.run("pyright --threads --skipunannotated --level warning")

    if rstlint:
        message_box('Linting "README.rst" file...')
        ctx.run("rst-lint README.rst")


@task
def api(ctx: Context, against: str = "develop", verbose: bool = False) -> None:
    """
    Check the public API for *semantic versioning* breakages with *Griffe*.

    Compare the working tree against a *Git* reference and report *breaking*
    changes only, e.g. removed public objects or altered signatures. Additions
    are backward-compatible and not reported. Fail on any breakage. Under
    *semantic versioning* such breakages require a compatibility version bump.

    Parameters
    ----------
    ctx
        Context.
    against
        *Git* reference (commit, branch, tag) to compare against.
    verbose
        Whether to expand each breakage into a detailed multi-line explanation.
    """

    message_box('Checking public API compatibility with "Griffe"...')
    ctx.run(
        f"griffe check {PYTHON_PACKAGE_NAME} --search . --against {against}"
        f"{' --format verbose' if verbose else ''}"
    )


@task
def prek(ctx: Context) -> None:
    """
    Run the "prek" hooks on the codebase.

    Parameters
    ----------
    ctx
        Context.
    """

    message_box('Running "prek" hooks on the codebase...')
    ctx.run("prek run --all-files")


@task
def tests(ctx: Context) -> None:
    """
    Run the unit tests with *Pytest*.

    Parameters
    ----------
    ctx
        Context.
    """

    message_box('Running "Pytest"...')
    ctx.run(
        "pytest "
        "--doctest-modules "
        f"--ignore={PYTHON_PACKAGE_NAME}/examples "
        f"--cov={PYTHON_PACKAGE_NAME} "
        f"{PYTHON_PACKAGE_NAME}"
    )


@task
def benchmark(ctx: Context, mode: str = "quick") -> None:
    """
    Run the cross-backend benchmarks.

    Parameters
    ----------
    ctx
        Context.
    mode
        Benchmark mode, either *quick* for a correctness smoke test or *full*
        for *HD* timing.
    """

    message_box("Running benchmarks...")
    ctx.run(f"python {os.path.join('utilities', 'benchmark.py')} --mode {mode}")


@task
def examples(ctx: Context, plots: bool = False) -> None:
    """
    Run the examples.

    Parameters
    ----------
    ctx
        Context.
    plots
        Whether to skip or only run the plotting examples: This a mutually
        exclusive switch.
    """

    message_box("Running examples...")

    for root, _dirnames, filenames in os.walk(
        os.path.join(PYTHON_PACKAGE_NAME, "examples")
    ):
        for filename in fnmatch.filter(filenames, "*.py"):
            if not plots and (
                "plotting" in root
                or "examples_contrast" in filename
                or "examples_hke" in filename
                or "examples_interpolation" in filename
                or "examples_sky" in filename
                or "examples_wilkie2021" in filename
            ):
                continue

            ctx.run(f"python {os.path.join(root, filename)}")


@task(formatting, quality, prek, tests, examples)
def preflight(ctx: Context) -> None:  # noqa: ARG001
    """
    Perform the preflight tasks.

    Parameters
    ----------
    ctx
        Context.
    """

    message_box('Finishing "Preflight"...')


@task
def docs(
    ctx: Context,
    plots: bool = True,
    html: bool = True,
    pdf: bool = True,
) -> None:
    """
    Build the documentation.

    Parameters
    ----------
    ctx
        Context.
    plots
        Whether to generate the documentation plots.
    html
        Whether to build the *HTML* documentation.
    pdf
        Whether to build the *PDF* documentation.
    """

    if plots:
        with ctx.cd("utilities"):
            message_box("Generating plots...")
            ctx.run("./generate_plots.py")

    with ctx.prefix("export COLOUR_SCIENCE__DOCUMENTATION_BUILD=True"), ctx.cd("docs"):
        if html:
            message_box('Building "HTML" documentation...')
            ctx.run("make html SPHINXOPTS='-W --keep-going'")

        if pdf:
            message_box('Building "PDF" documentation...')
            ctx.run("make latexpdf SPHINXOPTS='-W --keep-going'")


@task
def todo(ctx: Context) -> None:
    """
    Export the TODO items.

    Parameters
    ----------
    ctx
        Context.
    """

    message_box('Exporting "TODO" items...')

    with ctx.cd("utilities"):
        ctx.run("./export_todo.py")


@task
def requirements(ctx: Context) -> None:
    """
    Export the *requirements.txt* file.

    Parameters
    ----------
    ctx
        Context.
    """

    message_box('Exporting "requirements.txt" file...')
    ctx.run(
        "uv export --no-hashes --all-extras --no-emit-project "
        "--output-file requirements.txt"
    )

    message_box('Exporting "docs/requirements.txt" file...')
    ctx.run(
        "uv export --no-hashes --all-extras --no-dev --no-emit-project "
        "--output-file docs/requirements.txt"
    )


@task(literalise, clean, preflight, docs, todo, requirements)
def build(ctx: Context) -> None:
    """
    Build the project and runs dependency tasks.

    Parameters
    ----------
    ctx
        Context.
    """

    message_box("Building...")
    status = ctx.run("git status --porcelain -- README.rst", hide="both")
    if status is not None and status.stdout.strip():
        error = 'Please commit your changes to the "README.rst" file!'

        raise RuntimeError(error)

    readme_path = Path("README.rst")
    original_readme = readme_path.read_bytes()
    readme_content = (
        "..  image:: https://raw.githubusercontent.com/colour-science/"
        "colour-branding/master/images/Colour_Logo_001.png\n"
        + original_readme.decode("utf-8")
    )
    readme_content = re.sub(
        r"(\.\. begin-trim-long-description.*?\.\. end-trim-long-description)",
        "",
        readme_content,
        flags=re.DOTALL,
    )
    try:
        readme_path.write_text(readme_content, encoding="utf-8", newline="")
        ctx.run("uv build")
    finally:
        readme_path.write_bytes(original_readme)

    ctx.run("twine check dist/*.whl dist/*.tar.gz")


@task
def virtualise(ctx: Context, tests: bool = True) -> None:
    """
    Build a wheel from the source distribution and validate its installation.

    The wheel is installed in a fresh environment. Tests run outside the source
    tree with isolated Python imports, so an editable checkout cannot mask
    missing files or dependencies in the release artifacts.

    Parameters
    ----------
    ctx
        Context.
    tests
        Whether to run tests on the virtual environment.
    """

    archive = Path(
        "dist", f"{PYPI_ARCHIVE_NAME}-{APPLICATION_VERSION}.tar.gz"
    ).resolve()
    directory = archive.parent / f"{PYPI_PACKAGE_NAME}-{uuid.uuid4()}"
    directory.mkdir()
    with tarfile.open(archive) as distribution:
        distribution.extractall(directory, filter="data")
    project = directory / f"{PYPI_ARCHIVE_NAME}-{APPLICATION_VERSION}"
    wheels = directory / "wheels"
    ctx.run(f"uv build --wheel --out-dir {quote(str(wheels))} {quote(str(project))}")
    wheel = next(wheels.glob("*.whl"))

    environment = directory / ".venv"
    ctx.run(f"uv venv --python {quote(sys.executable)} {quote(str(environment))}")
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    interpreter = quote(str(python))
    configuration = tomllib.loads((project / "pyproject.toml").read_text())
    extras = ",".join(configuration["project"].get("optional-dependencies", {}))
    package = f"{wheel}[{extras}]" if extras else str(wheel)
    group = (
        f" --group {quote(str(project / 'pyproject.toml') + ':dev')}" if tests else ""
    )
    ctx.run(f"uv pip install --python {interpreter} {quote(package)}{group}")
    ctx.run(f"uv pip check --python {interpreter}")

    checks = directory / "checks"
    checks.mkdir()
    shutil.copyfile(project / "pyproject.toml", checks / "pyproject.toml")
    if (project / "conftest.py").exists():
        shutil.copyfile(project / "conftest.py", checks / "conftest.py")
    with ctx.cd(str(checks)):
        code = (
            f"import importlib.metadata; import {PYTHON_PACKAGE_NAME}; "
            f"assert importlib.metadata.version('{PYPI_PACKAGE_NAME}') "
            f"== '{APPLICATION_VERSION}'; print({PYTHON_PACKAGE_NAME}.__file__)"
        )
        result = ctx.run(f"{interpreter} -I -OO -c {quote(code)}", hide="both")
        if result is None:
            msg = "Could not locate the installed distribution."
            raise RuntimeError(msg)
        installed = Path(result.stdout.strip()).parent
        if not installed.is_relative_to(environment):
            msg = f"Package imported outside the validation environment: {installed}"
            raise RuntimeError(msg)
        ctx.run(
            f"{interpreter} -I -c "
            + quote("import imageio; imageio.plugins.freeimage.download()")
        )
        if tests:
            ctx.run(
                f"{interpreter} -I -m pytest --import-mode=append "
                f"--doctest-modules --ignore={quote(str(installed / 'examples'))} "
                f"--pyargs {PYTHON_PACKAGE_NAME}",
                env={"MPLBACKEND": "AGG", "OMP_NUM_THREADS": "1"},
            )


@task
def tag(ctx: Context, remote_name: str = "upstream") -> None:
    """
    Tag the repository according to defined version using *git-flow*.

    Parameters
    ----------
    ctx
        Context.
    remote_name
        Name of the remote repository.
    """

    message_box("Tagging...")
    result = ctx.run("git rev-parse --abbrev-ref HEAD", hide="both")

    if result is None or result.stdout.strip() != "develop":
        error = "Are you still on a feature or master branch?"

        raise RuntimeError(error)

    with open(os.path.join(PYTHON_PACKAGE_NAME, "__init__.py")) as file_handle:
        file_content = file_handle.read()
        major_version = re.search(
            '__major_version__\\s+=\\s+"(.*)"', file_content
        ).group(  # pyright: ignore
            1
        )
        minor_version = re.search(
            '__minor_version__\\s+=\\s+"(.*)"', file_content
        ).group(  # pyright: ignore
            1
        )
        change_version = re.search(
            '__change_version__\\s+=\\s+"(.*)"', file_content
        ).group(  # pyright: ignore
            1
        )

        version = f"{major_version}.{minor_version}.{change_version}"

        result = ctx.run(
            f"git ls-remote --tags --refs {quote(remote_name)}", hide="both"
        )
        if result is None:
            error = "Failed to list remote tags."

            raise RuntimeError(error)

        tags = {
            ref.removeprefix("refs/tags/").removesuffix("^{}")
            for line in result.stdout.splitlines()
            if (ref := line.partition("\t")[2]).startswith("refs/tags/")
        }
        if f"v{version}" in tags:
            error = (
                f'A "{PYTHON_PACKAGE_NAME}" "v{version}" tag already exists in '
                f"remote repository!"
            )

            raise RuntimeError(error)

        ctx.run(f"git flow release start v{version}")
        ctx.run(f"git flow release finish v{version}")


@task(build)
def release(ctx: Context) -> None:
    """
    Release the project to *Pypi* with *Twine*.

    Parameters
    ----------
    ctx
        Context.
    """

    message_box("Releasing...")
    with ctx.cd("dist"):
        ctx.run("twine upload *.tar.gz")
        ctx.run("twine upload *.whl")


@task
def sha256(ctx: Context) -> None:
    """
    Compute the project *Pypi* package *sha256* with *OpenSSL*.

    Parameters
    ----------
    ctx
        Context.
    """

    message_box('Computing "sha256"...')
    with ctx.cd("dist"):
        ctx.run(
            f"openssl sha256 {PYPI_ARCHIVE_NAME}-*.tar.gz {PYPI_ARCHIVE_NAME}-*.whl"
        )


@task(build, virtualise, sha256)
def pre_release(ctx: Context) -> None:  # noqa: ARG001
    """
    Validate the complete pre-release workflow without tagging or publishing.

    Run ``uv sync --all-extras`` followed by ``uv run invoke pre-release``.
    A current TeX installation is required for the HTML and PDF documentation.

    Parameters
    ----------
    ctx
        Context.
    """
    message_box("Pre-release validation complete.")
