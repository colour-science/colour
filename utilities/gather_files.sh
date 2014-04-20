#/bin/bash
echo -------------------------------------------------------------------------------
echo Color - Files Gathering
echo -------------------------------------------------------------------------------

export PROJECT=$( dirname "${BASH_SOURCE[0]}" )/..

export DOCUMENTATION=$PROJECT/docs/
export RELEASES=$PROJECT/releases/
export REPOSITORY=$RELEASES/repository/
export UTILITIES=$PROJECT/utilities

#! Color Changes gathering.
cp -rf $RELEASES/Changes.html $REPOSITORY/Color/

#! Color Manual / Help files.
cp -rf $DOCUMENTATION/help $REPOSITORY/Color/Help
rm $REPOSITORY/Foundations/help/Color_Manual.rst

#! Color Api files.
cp -rf $DOCUMENTATION/sphinx/build/html $REPOSITORY/Color/Api