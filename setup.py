from setuptools import setup, find_packages

setup(
    name='stanRepEx',
    version='0.1.0',
    description='Replica Exchange Monte Carlo using PyStan',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='narrowlyapplicable',
    url='https://github.com/narrowlyapplicable/stanRepEx',
    packages=find_packages(),
    install_requires=['numpy>1.13.0', 'pystan>2.19.0'],
    python_requires='>=3.6',

    classifiers=[
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Proggramming Language :: Python :: 3',
        'Proggramming Language :: Python :: 3.6',
        'Proggramming Language :: Python :: 3.7',
        'Proggramming Language :: Python :: 3.8',
        'Operating System :: OS Independent',
    ],
)