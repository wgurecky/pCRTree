from setuptools import setup, find_packages

setup(
    name = 'pCRTree',
    packages = find_packages(),
    author = 'William Gurecky',
    author_email = 'william.gurecky@utexas.edu',
    license = 'BSD-3',
    install_requires=['numpy>=1.8.0', 'scipy'],
)
