import os
from setuptools import setup, find_packages

THIS_REPO = "https://github.com/berenslab/fundus_image_toolbox"
ROOT = os.path.join(os.path.dirname(__file__))

exclude_strings = ["test", "egg", "0_", "1_", "outdated"]
SUBMODULES = [
    f
    for f in os.listdir(ROOT)
    if os.path.isdir(f)
    and not f.startswith(".")
    and not f.startswith("__")
    and not any([exclude_string in f for exclude_string in exclude_strings])
]

with open(os.path.join(ROOT, "Readme.md"), "r", encoding="utf8") as f:
    long_description = f.read()

with open(os.path.join(ROOT, "requirements.txt")) as f:
    required = f.read().splitlines()

# Add subfolders as dependencies
# TODO remove temporary branch
branch_name = "main"
submodules_required = []
# [
#     f"{submodule} @ git+{THIS_REPO}@{branch_name}#egg={submodule}&subdirectory={submodule}"
#     for submodule in SUBMODULES
# ]

# Include all subpackages.
# - as pip dependencies: use install_requires=required+submodules_required, packages=[], package_dir={}
# - from local subdirectories: use find_packages() and to prevent extra import
#   hierarchy, add an __init__.py into every outer subpackage folder and import the functions there.
#   Caveat: All python files in the outer subpackage folder will be imported, too, such as setup.py,
#   which is why we don't do this here.
setup(
    name="fundus_image_toolbox",
    version="0.0.1",
    author="Julius Gervelmeyer et al.",
    author_email="Julius.Gervelmeyer@uni-tuebingen.de",
    description=long_description.split("\n")[0],
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=THIS_REPO,
    packages=SUBMODULES,  # find_packages(),
    install_requires=required + submodules_required,
    # package_dir={"": ROOT},
    python_requires=">=3.9.0, <3.10",
    include_package_data=True,
)
