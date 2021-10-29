from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = ["numpy>=1.19.3","scipy>=1.7.1","statsmodels>=0.13.0","python-dateutil>=2.8.2","pytz>=2018.9","tzlocal>=1.5.1","regex>=2019.12.20","six>=1.15.0","dataparser>=0.0.1"]
#@a-x8j!dVWXm_Zi
setup(
    name="timexseries_c",
    version="0.0.11",
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