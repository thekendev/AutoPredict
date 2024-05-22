import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

__version__ = "0.1.0"
REPO_NAME = "End-to-End-Automotive-predictor-ML-project"
AUTHOR_USER_NAME = "thekendev"
SRC_REPO = "Autopredictor"
AUTHOR_EMAIL = "theken.dev@gmail.com"

setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="A small python package for ml app",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": "incomepredictor"},
    packages=setuptools.find_packages(where="Autopredictor"),
)