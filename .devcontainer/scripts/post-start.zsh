#!zsh
git config --global --add safe.directory /workspaces/DevContainer

poetry install

git fetch
git status -sb -uno
