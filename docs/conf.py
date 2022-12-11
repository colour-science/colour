"""
Colour - Documentation Configuration
====================================
"""

import os
import re
import setuptools.archive_util
import urllib.parse
import urllib.request

import colour as package  # noqa

basename = re.sub(
    "_(\\w)", lambda x: x.group(1).upper(), package.__name__.title()
)

if os.environ.get("READTHEDOCS") == "True":
    archive = "colour-plots.zip"
    branch = urllib.parse.quote(
        os.environ["READTHEDOCS_VERSION"]
        .replace("experimental-", "experimental/")
        .replace("feature-", "feature/")
        .replace("hotfix-", "hotfix/")
        .replace("latest", "develop")
        .replace("stable", "master"),
        safe="",
    )
    url = (
        f"https://nightly.link/colour-science/colour/workflows/"
        f"continuous-integration-documentation/{branch}/{archive}"
    )

    print(f"Using artifact url: {url}")

    urllib.request.urlretrieve(url, filename=archive)
    setuptools.archive_util.unpack_archive(archive, "_static")

# -- General configuration ------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.ifconfig",
    "sphinx.ext.inheritance_diagram",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinxcontrib.bibtex",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3.11", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/dev", None),
    "scipy": ("https://docs.scipy.org/doc/scipy-1.8.0/", None),
}

autodoc_member_order = "bysource"
autodoc_mock_imports = [
    "matplotlib",
    "matplotlib.cm",
    "matplotlib.image",
    "matplotlib.patches",
    "matplotlib.path",
    "matplotlib.pyplot",
    "matplotlib.ticker",
    "mpl_toolkits.mplot3d",
    "mpl_toolkits.mplot3d.art3d",
    "scipy",
    "scipy.interpolate",
    "scipy.ndimage",
    "scipy.optimize",
    "scipy.spatial",
    "scipy.spatial.distance",
]
autodoc_typehints = "both"
autodoc_type_aliases = {
    "ArrayLike": "ArrayLike",
    "DType": "DType",
    "DTypeBoolean": "DTypeBoolean",
    "DTypeComplex": "DTypeComplex",
    "DTypeFloat": "DTypeFloat",
    "DTypeInt": "DTypeInt",
    "DTypeReal": "DTypeReal",
    "Dataclass": "Dataclass",
    "NDArrayBoolean": "NDArrayBoolean",
    "NDArrayComplex": "NDArrayComplex",
    "NDArrayFloat": "NDArrayFloat",
    "NDArrayInt": "NDArrayInt",
    "NDArrayReal": "NDArrayReal",
    "NDArrayStr": "NDArrayStr",
    "Real": "Real",
}
autodoc_preserve_defaults = True

autoclass_content = "both"

autosummary_generate = True

bibtex_bibfiles = ["bibliography.bib"]
bibtex_encoding = "utf8"

napoleon_custom_sections = ["Attributes", "Methods"]

templates_path = ["_templates"]
source_suffix = ".rst"
master_doc = "index"

project = package.__application_name__
copyright = package.__copyright__.replace("Copyright (C)", "")
version = f"{package.__major_version__}.{package.__minor_version__}"
release = package.__version__

exclude_patterns = ["_build"]

pygments_style = "lovelace"

# -- Options for HTML output ----------------------------------------------
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "show_nav_level": 2,
    "icon_links": [
        {
            "name": "Email",
            "url": "mailto:colour-developers@colour-science.org",
            "icon": "fas fa-envelope",
        },
        {
            "name": "GitHub",
            "url": (
                f"https://github.com/colour-science/"
                f"{package.__name__.replace('_', '-')}"
            ),
            "icon": "fab fa-github",
        },
        {
            "name": "Facebook",
            "url": "https://www.facebook.com/python.colour.science",
            "icon": "fab fa-facebook",
        },
        {
            "name": "Gitter",
            "url": "https://gitter.im/colour-science/colour",
            "icon": "fab fa-gitter",
        },
        {
            "name": "Twitter",
            "url": "https://twitter.com/colour_science",
            "icon": "fab fa-twitter",
        },
    ],
}
html_logo = "_static/Logo_Light_001.svg"
html_static_path = ["_static"]
htmlhelp_basename = f"{basename}Doc"

# -- Options for LaTeX output ---------------------------------------------
latex_elements = {
    "papersize": "a4paper",
    "pointsize": "10pt",
    "preamble": """
\\usepackage{charter}
\\usepackage[defaultsans]{lato}
\\usepackage{inconsolata}

% Ignoring unicode errors.
\\makeatletter
\\def\\UTFviii@defined#1{%
    \\ifx#1\\relax
        ?%
    \\else\\expandafter
        #1%
    \\fi
}
\\makeatother
        """,
}
latex_documents = [
    (
        "index",
        f"{basename}.tex",
        f"{package.__application_name__} Documentation",
        package.__author__,
        "manual",
    ),
]
latex_logo = "_static/Logo_Medium_001.png"

# -- Options for manual page output ---------------------------------------
man_pages = [
    (
        "index",
        basename,
        f"{package.__application_name__} Documentation",
        [package.__author__],
        1,
    )
]

# -- Options for Texinfo output -------------------------------------------
texinfo_documents = [
    (
        "index",
        basename,
        f"{package.__application_name__} Documentation",
        package.__author__,
        package.__application_name__,
        basename,
        "Miscellaneous",
    ),
]

# -- Options for Epub output ----------------------------------------------
epub_title = package.__application_name__
epub_author = package.__author__
epub_publisher = package.__author__
epub_copyright = package.__copyright__.replace("Copyright (C)", "")
epub_exclude_files = ["search.html"]


def autodoc_process_docstring(app, what, name, obj, options, lines):
    """Process the docstrings to remove the *# noqa* *flake8* pragma."""

    for i, line in enumerate(lines):
        lines[i] = line.replace("# noqa", "")


def setup(app):
    """
    Prepare the extension and linking resources that Sphinx uses in the
    build process.
    """

    app.connect("autodoc-process-docstring", autodoc_process_docstring)
