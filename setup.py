from setuptools import setup


setup(
    name="su2tn",
    version="0.0.1",
    author="Victor Fischer",
    author_email="",
    packages=["su2tn",
              "classical_benchmark"],
    install_requires=[
        "numpy",
        "scipy",
        "sympy",
        "pandas",
        "networkx",
        "matplotlib"
    ],
)
