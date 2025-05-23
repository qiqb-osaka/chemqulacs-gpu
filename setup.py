from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# get the dependencies and installs
with open(path.join(here, "requirements.txt"), encoding="utf-8") as f:
    all_reqs = f.read().split("\n")

install_requires = [x.strip() for x in all_reqs if "git+" not in x]
dependency_links = [
    x.strip().replace("git+", "") for x in all_reqs if x.startswith("git+")
]


setup(
    name="vqe-qmmm",
    description="A set of tools to perfrom QM/MM with the VQE",
    packages=find_packages(exclude=["docs", "tests*"]),
    include_package_data=True,
    install_requires=install_requires,
    dependency_links=dependency_links,
    extras_require={"develop": ["black", "flake8", "docformatter", "pytest"]},
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
    test_suite="tests",
    python_requires=">=3.6",
)
