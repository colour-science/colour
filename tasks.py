# -*- coding: utf-8 -*-
"""
Invoke - Tasks
==============
"""

from __future__ import unicode_literals

import sys
if sys.version_info[:2] >= (3, 2):
    import biblib.bib
import fnmatch
import glob
import os
import re
import shutil
import tempfile
from invoke import task

import colour
from colour import read_image
from colour.utilities import message_box, metric_psnr

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2019 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'APPLICATION_NAME', 'PYTHON_PACKAGE_NAME', 'PYPI_PACKAGE_NAME',
    'BIBLIOGRAPHY_NAME', 'clean', 'formatting', 'tests', 'quality', 'examples',
    'docs', 'todo', 'preflight', 'build', 'virtualise', 'tag', 'release',
    'sha256'
]

APPLICATION_NAME = colour.__application_name__

PYTHON_PACKAGE_NAME = colour.__name__

PYPI_PACKAGE_NAME = 'colour-science'

BIBLIOGRAPHY_NAME = 'BIBLIOGRAPHY.bib'


@task
def clean(ctx, docs=True, bytecode=False):
    """
    Cleans the project.

    Parameters
    ----------
    ctx : invoke.context.Context
        Context.
    docs : bool, optional
        Whether to clean the *docs* directory.
    bytecode : bool, optional
        Whether to clean the bytecode files, e.g. *.pyc* files.

    Returns
    -------
    bool
        Task success.
    """
    message_box('Cleaning project...')

    patterns = ['build', '*.egg-info', 'dist']

    if docs:
        patterns.append('docs/_build')
        patterns.append('docs/generated')

    if bytecode:
        patterns.append('**/*.pyc')

    for pattern in patterns:
        ctx.run("rm -rf {}".format(pattern))


@task
def formatting(ctx, yapf=False, asciify=True, bibtex=True):
    """
    Formats the codebase with *Yapf* and converts unicode characters to ASCII.

    Parameters
    ----------
    ctx : invoke.context.Context
        Context.
    yapf : bool, optional
        Whether to format the codebase with *Yapf*.
    asciify : bool, optional
        Whether to convert unicode characters to ASCII.
    bibtex : bool, optional
        Whether to cleanup the *BibTeX* file.

    Returns
    -------
    bool
        Task success.
    """

    if yapf:
        message_box('Formatting codebase with "Yapf"...')
        ctx.run('yapf -p -i -r --exclude \'.git\' .')

    if asciify:
        message_box('Converting unicode characters to ASCII...')
        with ctx.cd('utilities'):
            ctx.run('./unicode_to_ascii.py')

    if bibtex and sys.version_info[:2] >= (3, 2):
        message_box('Cleaning up "BibTeX" file...')
        bibtex_path = BIBLIOGRAPHY_NAME
        with open(bibtex_path) as bibtex_file:
            bibtex = biblib.bib.Parser().parse(
                bibtex_file.read()).get_entries()

        for entry in sorted(bibtex.values(), key=lambda x: x.key):
            try:
                del entry['file']
            except KeyError:
                pass
            for key, value in entry.items():
                entry[key] = re.sub('(?<!\\\\)\\&', '\\&', value)

        with open(bibtex_path, 'w') as bibtex_file:
            for entry in bibtex.values():
                bibtex_file.write(entry.to_bib())
                bibtex_file.write('\n')


@task
def tests(ctx, nose=True):
    """
    Runs the unit tests with *Nose* or *Pytest*.

    Parameters
    ----------
    ctx : invoke.context.Context
        Context.
    nose : bool, optional
        Whether to use *Nose* or *Pytest*.

    Returns
    -------
    bool
        Task success.
    """

    if nose:
        message_box('Running "Nosetests"...')
        ctx.run(
            'nosetests --with-doctest --with-coverage --cover-package={0} {0}'.
            format(PYTHON_PACKAGE_NAME))
    else:
        message_box('Running "Pytest"...')
        ctx.run('pytest -W ignore')


@task
def quality(ctx, flake8=True, rstlint=True):
    """
    Checks the codebase with *Flake8* and lints various *restructuredText*
    files with *rst-lint*.

    Parameters
    ----------
    ctx : invoke.context.Context
        Context.
    flake8 : bool, optional
        Whether to check the codebase with *Flake8*.
    rstlint : bool, optional
        Whether to lint various *restructuredText* files with *rst-lint*.

    Returns
    -------
    bool
        Task success.
    """

    if flake8:
        message_box('Checking codebase with "Flake8"...')
        ctx.run('flake8 {0} --exclude=examples'.format(PYTHON_PACKAGE_NAME))

    if rstlint:
        message_box('Linting "README.rst" file...')
        ctx.run('rst-lint README.rst')


@task
def examples(ctx, plots=False):
    """
    Runs the examples.

    Parameters
    ----------
    ctx : invoke.context.Context
        Context.
    plots : bool, optional
        Whether to skip or only run the plotting examples: This a mutually
        exclusive switch.

    Returns
    -------
    bool
        Task success.
    """

    message_box('Running examples...')

    for root, _dirnames, filenames in os.walk(
            os.path.join(PYTHON_PACKAGE_NAME, 'examples')):
        for filename in fnmatch.filter(filenames, '*.py'):
            if not plots and ('plotting' in root or
                              'examples_interpolation' in filename or
                              'examples_contrast' in filename):
                continue

            if plots and ('plotting' not in root and
                          'examples_interpolation' not in filename and
                          'examples_contrast' not in filename):
                continue

            ctx.run('python {0}'.format(os.path.join(root, filename)))


@task
def docs(ctx, plots=True, html=True, pdf=True):
    """
    Builds the documentation.

    Parameters
    ----------
    ctx : invoke.context.Context
        Context.
    plots : bool, optional
        Whether to generate the documentation plots.
    html : bool, optional
        Whether to build the *HTML* documentation.
    pdf : bool, optional
        Whether to build the *PDF* documentation.

    Returns
    -------
    bool
        Task success.
    """

    if plots:
        temporary_directory = tempfile.mkdtemp()
        test_directory = os.path.join('docs', '_static')
        reference_directory = os.path.join(temporary_directory, '_static')
        try:
            shutil.copytree(test_directory, reference_directory)
            similar_plots = []
            with ctx.cd('utilities'):
                message_box('Generating plots...')
                ctx.run('./generate_plots.py')

                png_files = glob.glob('{0}/*.png'.format(reference_directory))
                for reference_png_file in png_files:
                    test_png_file = os.path.join(
                        test_directory, os.path.basename(reference_png_file))
                    psnr = metric_psnr(
                        read_image(str(reference_png_file))[::3, ::3],
                        read_image(str(test_png_file))[::3, ::3])
                    if psnr > 70:
                        similar_plots.append(test_png_file)
            with ctx.cd('docs/_static'):
                for similar_plot in similar_plots:
                    ctx.run('git checkout -- {0}'.format(
                        os.path.basename(similar_plot)))
        finally:
            shutil.rmtree(temporary_directory)

    with ctx.prefix('export COLOUR_SCIENCE_DOCUMENTATION_BUILD=True'):
        with ctx.cd('docs'):
            if html:
                message_box('Building "HTML" documentation...')
                ctx.run('make html')

            if pdf:
                message_box('Building "PDF" documentation...')
                ctx.run('make latexpdf')


@task
def todo(ctx):
    """
    Export the TODO items.

    Parameters
    ----------
    ctx : invoke.context.Context
        Context.

    Returns
    -------
    bool
        Task success.
    """

    message_box('Exporting "TODO" items...')

    with ctx.cd('utilities'):
        ctx.run('./export_todo.py')


@task(formatting, tests, quality, examples)
def preflight(ctx):
    """
    Performs the preflight tasks, i.e. *formatting*, *tests*, *quality*, and
    *examples*.

    Parameters
    ----------
    ctx : invoke.context.Context
        Context.

    Returns
    -------
    bool
        Task success.
    """

    message_box('Finishing "Preflight"...')


@task(docs, todo, preflight)
def build(ctx):
    """
    Builds the project and runs dependency tasks, i.e. *docs*, *todo*, and
    *preflight*.

    Parameters
    ----------
    ctx : invoke.context.Context
        Context.

    Returns
    -------
    bool
        Task success.
    """

    message_box('Building...')
    ctx.run('python setup.py sdist')
    ctx.run('python setup.py bdist_wheel --universal')


@task(clean, build)
def virtualise(ctx, tests=True):
    """
    Create a virtual environment for the project build.

    Parameters
    ----------
    ctx : invoke.context.Context
        Context.
    tests : bool, optional
        Whether to run tests on the virtual environment.

    Returns
    -------
    bool
        Task success.
    """

    pip_binary = '../staging/bin/pip'
    nosetests_binary = '../staging/bin/nosetests'

    with ctx.cd('dist'):
        ctx.run('tar -xvf {0}-*.tar.gz'.format(PYPI_PACKAGE_NAME))
        ctx.run('virtualenv staging')
        with ctx.cd('{0}-*'.format(PYPI_PACKAGE_NAME)):
            ctx.run('pwd')
            ctx.run('{0} install numpy'.format(pip_binary))
            ctx.run('{0} install -e .'.format(pip_binary))
            ctx.run('{0} install matplotlib'.format(pip_binary))
            ctx.run('{0} install nose'.format(pip_binary))
            ctx.run('{0} install mock'.format(pip_binary))
            if tests:
                ctx.run(nosetests_binary)


@task
def tag(ctx):
    """
    Tags the repository according to defined version using *git-flow*.

    Parameters
    ----------
    ctx : invoke.context.Context
        Context.

    Returns
    -------
    bool
        Task success.
    """

    message_box('Tagging...')
    result = ctx.run('git rev-parse --abbrev-ref HEAD', hide='both')
    assert result.stdout.strip() == 'develop', (
        'Are you still on a feature or master branch?')

    with open(os.path.join(PYTHON_PACKAGE_NAME, '__init__.py')) as file_handle:
        file_content = file_handle.read()
        major_version = re.search("__major_version__\s+=\s+'(.*)'",
                                  file_content).group(1)
        minor_version = re.search("__minor_version__\s+=\s+'(.*)'",
                                  file_content).group(1)
        change_version = re.search("__change_version__\s+=\s+'(.*)'",
                                   file_content).group(1)

        version = '.'.join((major_version, minor_version, change_version))

        result = ctx.run('git ls-remote --tags upstream', hide='both')
        remote_tags = result.stdout.strip().split('\n')
        tags = set()
        for remote_tag in remote_tags:
            tags.add(
                remote_tag.split('refs/tags/')[1].replace('refs/tags/', '^{}'))
        tags = sorted(list(tags))
        assert 'v{0}'.format(version) not in tags, (
            'A "{0}" "v{1}" tag already exists in remote repository!'.format(
                PYTHON_PACKAGE_NAME, version))

        ctx.run('git flow release start v{0}'.format(version))
        ctx.run('git flow release finish v{0}'.format(version))


@task(clean, build)
def release(ctx):
    """
    Releases the project to *Pypi* with *Twine*.

    Parameters
    ----------
    ctx : invoke.context.Context
        Context.

    Returns
    -------
    bool
        Task success.
    """

    message_box('Releasing...')
    with ctx.cd('dist'):
        ctx.run('twine upload *.tar.gz')
        ctx.run('twine upload *.whl')


@task
def sha256(ctx):
    """
    Computes the project *Pypi* package *sha256* with *OpenSSL*.

    Parameters
    ----------
    ctx : invoke.context.Context
        Context.

    Returns
    -------
    bool
        Task success.
    """

    message_box('Computing "sha256"...')
    with ctx.cd('dist'):
        ctx.run('openssl sha256 {0}-*.tar.gz'.format(PYPI_PACKAGE_NAME))
