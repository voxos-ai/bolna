#!/bin/bash

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print colored output
print_color() {
    printf "${1}${2}${NC}\n"
}

# Function to find the project root (where setup.py or pyproject.toml is located)
find_project_root() {
    local dir=$(pwd)
    while [[ "$dir" != "/" ]]; do
        if [[ -f "$dir/setup.py" || -f "$dir/pyproject.toml" ]]; then
            echo "$dir"
            return 0
        fi
        dir=$(dirname "$dir")
    done
    echo "Project root not found" >&2
    return 1
}

# Function to find and activate virtual environment
activate_venv() {
    local project_root=$1
    local venv_dir

    # Check common virtual environment directory names
    for venv in venv env .venv .env; do
        if [[ -f "$project_root/$venv/bin/activate" ]]; then
            venv_dir="$project_root/$venv"
            break
        fi
    done

    if [[ -n "$venv_dir" ]]; then
        print_color $YELLOW "Found virtual environment in $venv_dir"
        source "$venv_dir/bin/activate"
        print_color $GREEN "Virtual environment activated: $(which python)"
    else
        print_color $RED "No virtual environment found in the project root."
        exit 1
    fi
}

# Update the package
update_package() {
    local project_root=$(find_project_root)
    if [[ $? -ne 0 ]]; then
        print_color $RED "Failed to find project root. Are you in the correct directory?"
        exit 1
    fi

    print_color $GREEN "Project root found at: $project_root"
    cd "$project_root"

    print_color $GREEN "Activating virtual environment..."
    activate_venv "$project_root"

    print_color $GREEN "Updating package and dependencies..."

    # Update pip and setuptools
    pip install --upgrade pip setuptools

    # Install the package in editable mode
    if pip install -e .; then
        print_color $GREEN "Package updated successfully!"
    else
        print_color $RED "An error occurred while updating the package."
        exit 1
    fi

    # Check and install requirements if requirements.txt exists
    if [[ -f "requirements.txt" ]]; then
        print_color $GREEN "Installing requirements from requirements.txt..."
        if pip install -r requirements.txt; then
            print_color $GREEN "Requirements installed successfully!"
        else
            print_color $RED "An error occurred while installing the requirements."
            exit 1
        fi
    else
        print_color $YELLOW "No requirements.txt found. Skipping requirements installation."
    fi

}

# Main execution
main() {
    update_package
    print_color $GREEN "Update process completed successfully!"
}

# Run the main function
main