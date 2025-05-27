from setuptools import find_packages, setup
from typing import List

HYPHEN_e='-e .'

def get_requirements(file_path:str)->List[str]:
    '''this func will return the list of requirements'''
    requirements=[]
    with open(file_path) as f:
        requirements=[line.strip() for line in f if line.strip()]

        if HYPHEN_e in requirements:
            requirements.remove(HYPHEN_e)
    
    return requirements


setup(
    name='studentPerformanceIndicator', 
    version='0.0.1', 
    author='hrish',
    author_email='hrishkul@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)