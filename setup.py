from setuptools import setup, find_packages

setup(
    name = 'pCRTree',
    packages = find_packages(),
    author = 'William Gurecky',
    author_email = 'william.gurecky@utexas.edu',
    license = 'MIT',
    install_requires=['numpy>=1.8.0', 'scipy'],
)
