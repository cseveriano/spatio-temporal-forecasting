from setuptools import setup, find_packages

setup(
    name='spatio-temporal-forecasting',
    packages=['spatiotemporal', 'spatiotemporal.models', 'spatiotemporal.models.clusteredmvfts','spatiotemporal.models.clusteredmvfts.fts', 'spatiotemporal.models.clusteredmvfts.partitioner', 'spatiotemporal.data',
	'spatiotemporal.models.benchmarks', 'spatiotemporal.models.benchmarks.fbem',
	'spatiotemporal.models.benchmarks.granularfts','spatiotemporal.models.benchmarks.mvfts',
	'spatiotemporal.features', 'spatiotemporal.test', 'spatiotemporal.util', 'spatiotemporal.visualization'],
	package_dir={'spatiotemporal': 'src/spatiotemporal'},
    version='1.0',
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