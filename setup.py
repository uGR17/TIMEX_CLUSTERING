from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = ["dateparser<2.0.0,>=1.0.0",\
    "torch>=1.7.0",\
    "tslearn>=0.5.0",\
    "matplotlib>=2.0.0",\
    "wheel>=0.33.0",\
    "colorhash<2.0.0,>=1.0.3",\
    "networkx>=2.5.0",\
    "dash<2.0.0,>=1.19.0",\
    "dash-bootstrap-components>=0.11.3",\
    "pmdarima<2.0.0,>=1.8.2",\
    "gunicorn<21.0.0,>=20.0.4",\
    "sklearn>=0.0,<0.1",\
    "scipy<2.0.0,>=1.6.0",\
    "statsmodels<0.13.0,>=0.12.2",\
    "pytest>=4.3.0"]

setup(
    name="timexseries_clustering",
    version="0.0.45",
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