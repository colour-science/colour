#!/bin/bash
echo -------------------------------------------------------------------------------
echo Sphinx Documentation Build
echo -------------------------------------------------------------------------------

while getopts rhp OPTION
do
   case "$OPTION" in
      r) OPTION_REMOVE_BUILD_AND_SOURCE_FILES=$OPTARG;;
      h) OPTION_GENERATE_HTML_FILES=$OPTARG;;
      p) OPTION_GENERATE_PDF_FILE=$OPTARG;;
   esac
done

export PROJECT_NAME=colour
export PROJECT_DIRECTORY=$( dirname "${BASH_SOURCE[0]}" )/..

export READTHEDOCS=True

export UTILITIES_DIRECTORY=$PROJECT_DIRECTORY/utilities
export PACKAGE_DIRECTORY=$PROJECT_DIRECTORY/$PROJECT_NAME
export SPHINX_DOCUMENTATION_DIRECTORY=$PROJECT_DIRECTORY/docs
export SPHINX_DOCUMENTATION_BUILD_DIRECTORY=$SPHINX_DOCUMENTATION_DIRECTORY/_build

#! Removing previous build elements.
if [ -n "${OPTION_REMOVE_BUILD_AND_SOURCE_FILES+1}" ]; then
    echo -------------------------------------------------------------------------------
    echo API Build and Source Files Removal - Begin
    echo -------------------------------------------------------------------------------
    rm -rfv $SPHINX_DOCUMENTATION_BUILD_DIRECTORY
    rm -rfv $SPHINX_DOCUMENTATION_DIRECTORY/generated
    echo -------------------------------------------------------------------------------
    echo API Build and Source Files Removal - End
    echo -------------------------------------------------------------------------------
fi

#! Building the HTML documentation.
if [ -n "${OPTION_GENERATE_HTML_FILES+1}" ]; then
    echo -------------------------------------------------------------------------------
    echo Sphinx Documentation Build - Begin
    echo -------------------------------------------------------------------------------
    cd $SPHINX_DOCUMENTATION_DIRECTORY
    make html
    echo -------------------------------------------------------------------------------
    echo Sphinx Documentation Build - End
    echo -------------------------------------------------------------------------------
fi

#! Building the PDF documentation.
if [ -n "${OPTION_GENERATE_PDF_FILE+1}" ]; then
    echo -------------------------------------------------------------------------------
    echo Sphinx Documentation Build - Begin
    echo -------------------------------------------------------------------------------
    cd $SPHINX_DOCUMENTATION_DIRECTORY
    make latexpdf
    echo -------------------------------------------------------------------------------
    echo Sphinx Documentation Build - End
    echo -------------------------------------------------------------------------------
fi
