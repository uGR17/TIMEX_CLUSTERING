from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = ["numpy>=1.19.3"]

setup(
    name="timexseries_c",
    version="0.0.4",
    author="Uriel Guadarrama Ramirez",
    author_email="u.guadarrama@hotmail.com",
    description="TIMEX-CLUSTERING is a framework for time-series-clustering-as-a-service",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/uGR17/TIMEX_CLUSTERING",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Information Analysis"
    ],
)