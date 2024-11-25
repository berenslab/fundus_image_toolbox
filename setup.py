from setuptools import setup, find_packages

setup(
    name="fundus_image_toolbox",
    version="0.0.1",
    author="Julius Gervelmeyer et al.",
    author_email="Julius.Gervelmeyer@uni-tuebingen.de",
    description="A toolbox for fundus image processing",
    long_description=open("Readme.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=open("requirements.txt").read().splitlines(),
    python_requires=">=3.9.0, <3.10",
    include_package_data=True,
)
