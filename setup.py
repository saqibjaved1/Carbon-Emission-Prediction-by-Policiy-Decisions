from setuptools import setup, find_packages
setup(
    name='predictCO2',
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    version="1.0",
    description="Group7-AMI",
    install_requires=open("./requirements.txt").readline()
)