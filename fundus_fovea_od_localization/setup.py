import os
from setuptools import setup, find_packages

project = os.path.basename(os.path.dirname(os.path.realpath(__file__)))
# about = __import__(project).about

with open("Readme.md", "r", encoding='utf8') as f:
    long_description = f.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name=project,
    version="0.0.1",
    author="Julius Gervelmeyer et al.",
    author_email="Julius.Gervelmeyer@uni-tuebingen.de",
    description=long_description.split('\n')[0],
    long_description=long_description,
    long_description_content_type="text/markdown",
    # url=None,
    packages=find_packages(),
    install_requires=required,
    package_dir={'': '.'},
    python_requires='>=3.9.0, <3.10',
    include_package_data=True,
    package_data={
        '': ['*.csv'],
        }
)