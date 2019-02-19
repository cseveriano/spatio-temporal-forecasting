from setuptools import setup, find_packages

setup(
    name='spatio-temporal-forecasting',
    package_dir={'':'src/models'},
    packages=find_packages("src/models"),
    version='0.1',
    description='Spatio Temporal Forecasting',
    author='Carlos Severiano',
    author_email='carlossjr@gmail.com',
    url='https://github.com/cseveriano/spatio-temporal-forecasting',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
)