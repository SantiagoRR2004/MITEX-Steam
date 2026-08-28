#!/bin/bash

# Current script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

REPO_URL="https://github.com/SantiagoRR2004/Steam-TopSellers.git"
REPO_DIR="$SCRIPT_DIR/TopSellers"

if [ -d "$REPO_DIR" ]; then
    echo "$REPO_URL exists, pulling latest changes..."
    (
        cd "$REPO_DIR" || exit
        git pull
    )

else
    echo "$REPO_URL does not exist, cloning repository..."
    git clone "$REPO_URL" "$REPO_DIR"

fi