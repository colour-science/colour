#!/bin/bash
echo -------------------------------------------------------------------------------
echo Sphinx Documentation Build
echo -------------------------------------------------------------------------------

while getopts ram OPTION
do
   case "$OPTION" in
      r) OPTION_REMOVE_BUILD_AND_SOURCE_FILES=$OPTARG;;
      a) OPTION_GENERATE_API_FILES=$OPTARG;;
      m) OPTION_GENERATE_HTML_FILES=$OPTARG;;
   esac
done

export PROJECT_NAME=colour
export PROJECT_DIRECTORY=$( dirname "${BASH_SOURCE[0]}" )/..

export UTILITIES_DIRECTORY=$PROJECT_DIRECTORY/utilities
export PACKAGE_DIRECTORY=$PROJECT_DIRECTORY/$PROJECT_NAME
export SPHINX_DOCUMENTATION_DIRECTORY=$PROJECT_DIRECTORY/docs
export SPHINX_DOCUMENTATION_BUILD_DIRECTORY=$SPHINX_DOCUMENTATION_DIRECTORY/_build

#! Removing previous build elements.
if [ -n "${OPTION_REMOVE_BUILD_AND_SOURCE_FILES+1}" ]; then
    echo -------------------------------------------------------------------------------
    echo API Build and Source Files Removal - Begin
    echo -------------------------------------------------------------------------------
    rm -rfv $SPHINX_DOCUMENTATION_BUILD_DIRECTORY/doctrees
    rm -rfv $SPHINX_DOCUMENTATION_BUILD_DIRECTORY/html
    rm -fv $SPHINX_DOCUMENTATION_DIRECTORY/$PROJECT_NAME*.rst
    rm -fv $SPHINX_DOCUMENTATION_DIRECTORY/modules.rst
    echo -------------------------------------------------------------------------------
    echo API Build and Source Files Removal - End
    echo -------------------------------------------------------------------------------
fi

#! Generating the API files.
if [ -n "${OPTION_GENERATE_API_FILES+1}" ]; then
    echo -------------------------------------------------------------------------------
    echo API Files Generation - Begin
    echo -------------------------------------------------------------------------------
    #! Filtering tests modules.
    export EXCLUDED_MODULES=$( find "${PACKAGE_DIRECTORY}" -name '*tests*' | xargs )
    sphinx-apidoc -fe -o $SPHINX_DOCUMENTATION_DIRECTORY $PACKAGE_DIRECTORY $EXCLUDED_MODULES
    cd $SPHINX_DOCUMENTATION_DIRECTORY
    sed -i 's/module$/Module/g; s/package$/Package/g; s/^-----------$/------------/g; s/^----------$/-----------/g; s/^Subpackages$/Sub-Packages/g; s/^Submodules$/Sub-Modules/g; s/^Module contents$/Module Contents/g' $PROJECT_NAME*.rst
    echo -------------------------------------------------------------------------------
    echo API Files Generation - End
    echo -------------------------------------------------------------------------------
fi

#! Building the documentation.
if [ -n "${OPTION_GENERATE_HTML_FILES+1}" ]; then
    echo -------------------------------------------------------------------------------
    echo Sphinx Documentation Build - Begin
    echo -------------------------------------------------------------------------------
    cd $SPHINX_DOCUMENTATION_DIRECTORY
    make html 2>&1 | grep -v --line-buffered -e "Duplicate target name, cannot be used as a unique reference" -e "more than one target found for cross-reference" -e "Duplicate explicit target name"
    echo -------------------------------------------------------------------------------
    echo Sphinx Documentation Build - End
    echo -------------------------------------------------------------------------------
fi
