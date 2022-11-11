"""
Invoke - Tasks
==============
"""

from __future__ import annotations

import biblib.bib
import fnmatch
import os
import re
import toml
import uuid

import colour
from colour.hints import Boolean
from colour.utilities import message_box

import inspect

if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec

from invoke import Context, task


__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "APPLICATION_NAME",
    "APPLICATION_VERSION",
    "PYTHON_PACKAGE_NAME",
    "PYPI_PACKAGE_NAME",
    "BIBLIOGRAPHY_NAME",
    "clean",
    "formatting",
    "quality",
    "precommit",
    "tests",
    "examples",
    "preflight",
    "docs",
    "todo",
    "requirements",
    "build",
    "virtualise",
    "tag",
    "release",
    "sha256",
]

APPLICATION_NAME: str = colour.__application_name__

APPLICATION_VERSION: str = colour.__version__

PYTHON_PACKAGE_NAME: str = colour.__name__

PYPI_PACKAGE_NAME: str = "colour-science"

BIBLIOGRAPHY_NAME: str = "BIBLIOGRAPHY.bib"


@task
def clean(
    ctx: Context,
    docs: Boolean = True,
    bytecode: Boolean = False,
    mypy: Boolean = True,
    pytest: Boolean = True,
):
    """
    Clean the project.

    Parameters
    ----------
    ctx
        Context.
    docs
        Whether to clean the *docs* directory.
    bytecode
        Whether to clean the bytecode files, e.g. *.pyc* files.
    mypy
        Whether to clean the *Mypy* cache directory.
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

    if mypy:
        patterns.append(".mypy_cache")

    if pytest:
        patterns.append(".pytest_cache")

    for pattern in patterns:
        ctx.run(f"rm -rf {pattern}")


@task
def formatting(
    ctx: Context,
    asciify: Boolean = True,
    bibtex: Boolean = True,
):
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
            entries = (
                biblib.bib.Parser().parse(bibtex_file.read()).get_entries()
            )

        for entry in sorted(entries.values(), key=lambda x: x.key):
            try:
                del entry["file"]
            except KeyError:
                pass

            for key, value in entry.items():
                entry[key] = re.sub("(?<!\\\\)\\&", "\\&", value)

        with open(bibtex_path, "w") as bibtex_file:
            for entry in sorted(entries.values(), key=lambda x: x.key):
                bibtex_file.write(entry.to_bib())
                bibtex_file.write("\n")


@task
def quality(
    ctx: Context,
    mypy: Boolean = True,
    rstlint: Boolean = True,
):
    """
    Check the codebase with *Mypy* and lints various *restructuredText*
    files with *rst-lint*.

    Parameters
    ----------
    ctx
        Context.
    flake8
        Whether to check the codebase with *Flake8*.
    mypy
        Whether to check the codebase with *Mypy*.
    rstlint
        Whether to lint various *restructuredText* files with *rst-lint*.
    """

    if mypy:
        message_box('Checking codebase with "Mypy"...')
        ctx.run(
            f"mypy "
            f"--install-types "
            f"--non-interactive "
            f"--show-error-codes "
            f"--warn-unused-ignores "
            f"--warn-redundant-casts "
            f"-p {PYTHON_PACKAGE_NAME} "
            f"|| true"
        )

    if rstlint:
        message_box('Linting "README.rst" file...')
        ctx.run("rst-lint README.rst")


@task
def precommit(ctx: Context):
    """
    Run the "pre-commit" hooks on the codebase.

    Parameters
    ----------
    ctx
        Context.
    """

    message_box('Running "pre-commit" hooks on the codebase...')
    ctx.run("pre-commit run --all-files")


@task
def tests(ctx: Context):
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
        "--disable-warnings "
        "--doctest-modules "
        f"--ignore={PYTHON_PACKAGE_NAME}/examples "
        f"{PYTHON_PACKAGE_NAME}"
    )


@task
def examples(ctx: Context, plots: Boolean = False):
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
            ):
                continue

            ctx.run(f"python {os.path.join(root, filename)}")


@task(formatting, quality, precommit, tests, examples)
def preflight(ctx: Context):
    """
    Perform the preflight tasks, i.e. *formatting*, *tests*, *quality*, and
    *examples*.

    Parameters
    ----------
    ctx
        Context.
    """

    message_box('Finishing "Preflight"...')


@task
def docs(
    ctx: Context,
    plots: Boolean = True,
    html: Boolean = True,
    pdf: Boolean = True,
):
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

    with ctx.prefix("export COLOUR_SCIENCE__DOCUMENTATION_BUILD=True"):
        with ctx.cd("docs"):
            if html:
                message_box('Building "HTML" documentation...')
                ctx.run("make html")

            if pdf:
                message_box('Building "PDF" documentation...')
                ctx.run("make latexpdf")


@task
def todo(ctx: Context):
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
def requirements(ctx: Context):
    """
    Export the *requirements.txt* file.

    Parameters
    ----------
    ctx
        Context.
    """

    message_box('Exporting "requirements.txt" file...')
    ctx.run(
        "poetry run pip list --format=freeze | "
        'egrep -v "colour==|colour-science==" '
        "> requirements.txt"
    )


@task(clean, preflight, docs, todo, requirements)
def build(ctx: Context):
    """
    Build the project and runs dependency tasks, i.e. *docs*, *todo*, and
    *preflight*.

    Parameters
    ----------
    ctx
        Context.
    """

    message_box("Building...")
    if "modified:   pyproject.toml" in ctx.run("git status").stdout:
        raise RuntimeError(
            'Please commit your changes to the "pyproject.toml" file!'
        )

    pyproject_content = toml.load("pyproject.toml")
    pyproject_content["tool"]["poetry"]["name"] = PYPI_PACKAGE_NAME
    pyproject_content["tool"]["poetry"]["packages"] = [
        {"include": PYTHON_PACKAGE_NAME, "from": "."}
    ]
    with open("pyproject.toml", "w") as pyproject_file:
        toml.dump(pyproject_content, pyproject_file)

    if "modified:   README.rst" in ctx.run("git status").stdout:
        raise RuntimeError(
            'Please commit your changes to the "README.rst" file!'
        )

    with open("README.rst") as readme_file:
        readme_content = readme_file.read()

    with open("README.rst", "w") as readme_file:
        readme_file.write(
            re.sub(
                (
                    "(\\.\\. begin-trim-long-description.*?"
                    "\\.\\. end-trim-long-description)"
                ),
                "",
                readme_content,
                flags=re.DOTALL,
            )
        )

    ctx.run("poetry build")
    ctx.run("git checkout -- pyproject.toml")
    ctx.run("git checkout -- README.rst")

    with ctx.cd("dist"):
        ctx.run(f"tar -xvf {PYPI_PACKAGE_NAME}-{APPLICATION_VERSION}.tar.gz")
        ctx.run(f"cp {PYPI_PACKAGE_NAME}-{APPLICATION_VERSION}/setup.py ../")

        ctx.run(f"rm -rf {PYPI_PACKAGE_NAME}-{APPLICATION_VERSION}")

    with open("setup.py") as setup_file:
        source = setup_file.read()

    setup_kwargs = []

    def sub_callable(match):
        setup_kwargs.append(match)

        return ""

    template = """
setup({0}
)
"""

    source = re.sub(
        "from setuptools import setup",
        (
            '"""\n'
            "Colour - Setup\n"
            "==============\n"
            '"""\n\n'
            "import codecs\n"
            "from setuptools import setup"
        ),
        source,
    )
    source = re.sub(
        "setup_kwargs = {(.*)}.*setup\\(\\*\\*setup_kwargs\\)",
        sub_callable,
        source,
        flags=re.DOTALL,
    )[:-2]
    setup_kwargs = setup_kwargs[0].group(1).splitlines()
    for i, line in enumerate(setup_kwargs):
        setup_kwargs[i] = re.sub("^\\s*('(\\w+)':\\s?)", "    \\2=", line)
        if setup_kwargs[i].strip().startswith("long_description"):
            setup_kwargs[i] = (
                "    long_description="
                "codecs.open('README.rst', encoding='utf8')"
                ".read(),"
            )

    source += template.format("\n".join(setup_kwargs))

    with open("setup.py", "w") as setup_file:
        setup_file.write(source)

    ctx.run("poetry run pre-commit run --files setup.py || true")

    ctx.run("twine check dist/*")


@task
def virtualise(ctx: Context, tests: Boolean = True):
    """
    Create a virtual environment for the project build.

    Parameters
    ----------
    ctx
        Context.
    tests
        Whether to run tests on the virtual environment.
    """

    unique_name = f"{PYPI_PACKAGE_NAME}-{uuid.uuid1()}"
    with ctx.cd("dist"):
        ctx.run(f"tar -xvf {PYPI_PACKAGE_NAME}-{APPLICATION_VERSION}.tar.gz")
        ctx.run(f"mv {PYPI_PACKAGE_NAME}-{APPLICATION_VERSION} {unique_name}")
        with ctx.cd(unique_name):
            ctx.run(
                'poetry install --extras "graphviz meshing optional plotting"'
            )
            ctx.run("source $(poetry env info -p)/bin/activate")
            ctx.run(
                'python -c "import imageio;'
                'imageio.plugins.freeimage.download()"'
            )
            if tests:
                ctx.run(
                    "poetry run py.test "
                    "--disable-warnings "
                    "--doctest-modules "
                    f"--ignore={PYTHON_PACKAGE_NAME}/examples "
                    f"{PYTHON_PACKAGE_NAME}",
                    env={"MPLBACKEND": "AGG"},
                )


@task
def tag(ctx: Context):
    """
    Tag the repository according to defined version using *git-flow*.

    Parameters
    ----------
    ctx
        Context.
    """

    message_box("Tagging...")
    result = ctx.run("git rev-parse --abbrev-ref HEAD", hide="both")

    assert (
        result.stdout.strip() == "develop"
    ), "Are you still on a feature or master branch?"

    with open(os.path.join(PYTHON_PACKAGE_NAME, "__init__.py")) as file_handle:
        file_content = file_handle.read()
        major_version = re.search(
            '__major_version__\\s+=\\s+"(.*)"', file_content
        ).group(  # type: ignore[union-attr]
            1
        )
        minor_version = re.search(
            '__minor_version__\\s+=\\s+"(.*)"', file_content
        ).group(  # type: ignore[union-attr]
            1
        )
        change_version = re.search(
            '__change_version__\\s+=\\s+"(.*)"', file_content
        ).group(  # type: ignore[union-attr]
            1
        )

        version = ".".join((major_version, minor_version, change_version))

        result = ctx.run("git ls-remote --tags upstream", hide="both")
        remote_tags = result.stdout.strip().split("\n")
        tags = set()
        for remote_tag in remote_tags:
            tags.add(
                remote_tag.split("refs/tags/")[1].replace("refs/tags/", "^{}")
            )
        version_tags = sorted(list(tags))
        assert f"v{version}" not in version_tags, (
            f'A "{PYTHON_PACKAGE_NAME}" "v{version}" tag already exists in '
            f"remote repository!"
        )

        ctx.run(f"git flow release start v{version}")
        ctx.run(f"git flow release finish v{version}")


@task(build)
def release(ctx: Context):
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
def sha256(ctx: Context):
    """
    Compute the project *Pypi* package *sha256* with *OpenSSL*.

    Parameters
    ----------
    ctx
        Context.
    """

    message_box('Computing "sha256"...')
    with ctx.cd("dist"):
        ctx.run(f"openssl sha256 {PYPI_PACKAGE_NAME}-*.tar.gz")
