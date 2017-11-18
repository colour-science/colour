#!/bin/bash
echo -------------------------------------------------------------------------------
echo Colour - Run Examples Begin
echo -------------------------------------------------------------------------------

export PROJECT=$( dirname "${BASH_SOURCE[0]}" )/..

export EXAMPLES=$PROJECT/colour/examples/

for i in $(find "${EXAMPLES}" -name \*.py); do
    python "$i"
    if [ $? -eq 1 ]; then
        exit
    fi
    echo ""
    read -t 5 -p "Press 'ENTER' to continue."
    echo ""
done

echo -------------------------------------------------------------------------------
echo Colour - Run Examples End
echo -------------------------------------------------------------------------------
