from setuptools import setup, find_packages

setup(
    name='aeolia',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A PyTorch Geometric Temporal project for graph neural networks and temporal data analysis.',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'torch',
        'torch-geometric',
        'numpy',
        'pandas',
        'matplotlib',
        'scikit-learn',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)