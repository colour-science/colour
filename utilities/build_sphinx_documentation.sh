#!/bin/bash
echo -------------------------------------------------------------------------------
echo Colour - Sphinx Documentation Build
echo -------------------------------------------------------------------------------

while getopts ra OPTION
do
   case "$OPTION" in
      r) OPTION_REMOVE_BUILD_AND_SOURCE_FILES=$OPTARG;;
      a) OPTION_GENERATE_API_FILES=$OPTARG;;
   esac
done

export PROJECT=$( dirname "${BASH_SOURCE[0]}" )/..

export UTILITIES=$PROJECT/utilities
export PACKAGE=$PROJECT/colour
export SPHINX_DOCUMENTATION=$PROJECT/docs
export SPHINX_DOCUMENTATION_BUILD=$SPHINX_DOCUMENTATION/_build

#! Removing previous build elements.
if [ -n "${OPTION_REMOVE_BUILD_AND_SOURCE_FILES+1}" ]; then
    echo -------------------------------------------------------------------------------
    echo API Build and Source Files Removal - Begin
    echo -------------------------------------------------------------------------------
    rm -rfv $SPHINX_DOCUMENTATION_BUILD/doctrees
    rm -rfv $SPHINX_DOCUMENTATION_BUILD/html
    rm -fv $SPHINX_DOCUMENTATION/colour*.rst
    rm -fv $SPHINX_DOCUMENTATION/modules.rst
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
    export EXCLUDED_MODULES=$( find "${PACKAGE}" -name '*tests*' | xargs )
    python $UTILITIES/libraries/python/apidoc.py -fe -o $SPHINX_DOCUMENTATION $PACKAGE $EXCLUDED_MODULES
    echo -------------------------------------------------------------------------------
    echo API Files Generation - End
    echo -------------------------------------------------------------------------------
fi

#! Building the documentation.
echo -------------------------------------------------------------------------------
echo Sphinx Documentation Build - Begin
echo -------------------------------------------------------------------------------
cd $SPHINX_DOCUMENTATION
make html
echo -------------------------------------------------------------------------------
echo Sphinx Documentation Build - End
echo -------------------------------------------------------------------------------
