from setuptools import setup, find_packages
setup(
    name='predictCO2',
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    version="1.0",
    description="Background Processing Server",
    author="sebastian.lettner24@gmail.com",
    install_requires=open("./requirements.txt").readline()
)