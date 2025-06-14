"""
setup.py is the traditional file used in Python packaging.

It defines metadata about the project (name, version, author, etc.)
and installation behavior (e.g., dependencies, included packages).

While modern projects use pyproject.toml, many existing tools and CI/CD
pipelines still support or require setup.py — so it's still seen in practice.
"""

from setuptools import find_packages, setup
from typing import List

def get_requirements() -> List[str]:
    """
    Reads the project's requirements.txt file and returns a clean list
    of dependencies (excluding empty lines or '-e .' used for editable installs).

    Returns:
        List[str]: A list of Python packages to install.
    """
    requirement_list: List[str] = []
    try:
        with open("requirements.txt", "r") as file:
            lines = file.readlines()
            for line in lines:
                # Remove whitespace and newlines
                requirement = line.strip()

                # Exclude empty lines and editable flag (-e .)
                if requirement and requirement != "-e .":
                    requirement_list.append(requirement)

    except FileNotFoundError:
        print("requirements.txt file not found. Skipping install_requires.")

    return requirement_list

# Standard setuptools setup call
setup(
    name="NetworkworkSecurity",                 # Project name
    version="0.0.1",                            # Initial version
    author="Gabriel Adebayo",                   # Author name
    author_email="iyanuoluwaadebayo04@gmail.com",  # Contact
    packages=find_packages(),                   # Auto-detect all packages (__init__.py)
    install_requires=get_requirements(),        # Read from requirements.txt
)
