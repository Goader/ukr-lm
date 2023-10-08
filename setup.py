from setuptools import setup, find_packages


with open('requirements.txt') as f:
    required = f.read().splitlines()


setup(
    name='ukrlm',
    version='0.0.1',
    packages=find_packages(include=['ukrlm', 'ukrlm.*']),
    install_requires=required,
)
