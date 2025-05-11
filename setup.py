"""
LiteDetective - Malicious Content Detection Pipeline

Copyright (c) 2025 Albert Zhao
Author: Albert Zhao Zhaoq@kean.edu Hu Mingcheng 
Created: 2025-05-11
Updated: 2025-05-11

Description:
    Setup script for the package.

License: MIT License
"""

from setuptools import setup, find_packages

setup(
    name='litedetective',
    version='0.1.0',
    author='Albert Zhao',
    author_email='zhaoq@kean.edu',
    description='A lightweight Chinese malicious comment detection pipeline',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/AlbertZhaoCA/litedetective',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'torch',
        'transformers',
        'pandas',
        'scikit-learn',
        'jieba',
        'numpy',
        'tqdm',
        'openai==1.76.2',
        'python-dotenv',
        'tenacity',
        'aiohttp',
        'pydantic',
        'typing-extensions'
    ],
    python_requires='>=3.8',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
        'License :: OSI Approved :: MIT License',
    ],
    entry_points={
        'console_scripts': [
            'build-policy=libs.data_generator.policy:main',
            'build-train-data=libs.data_generator.train_data:main'
        ],
    },
)
