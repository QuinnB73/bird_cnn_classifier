from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['google-cloud-storage>=1.19', 'numpy>=1.16']

setup(
    name='trainer',
    version='0.1',
    install_packages=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='Custom training application package'
)

