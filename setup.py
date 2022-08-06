from setuptools import find_packages, setup

version = "0.1.0"

setup(
    name='flexnets',
    version=version,
    author='Vladyslav Branytskyi, Diana Malyk',
    author_email='vladyslav.branytskyi@nure.ua, diana.malyk@nure.ua',
    description='Hyper-Flexible Convolutional Neural Networks Based on Generalized Lehmer Mean',
    url='https://github.com/Adversarial-Intelligence-Group/flexnets',
    license='MIT',
    packages=find_packages(),
    python_requires='>=3.7',
    keywords=['deep learning', 'machine learning',
              'pooling', 'convolutional neural network']
)
