from setuptools import setup, find_packages

__version__ = '0.1.3'
url = 'https://github.com/IsaacCorley/pytorch_enhance'

install_requires = [
    'torch',
    'torchvision',
    'pillow'
]
setup_requires = ['pytest-runner']
tests_require = ['pytest', 'pytest-cov', 'mock']

setup(
    name='torch_enhance',
    packages=find_packages(),
    version=__version__,
    license='Apache License 2.0',
    description='Image Super-Resolution Library for PyTorch',
    author='Isaac Corley',
    author_email='isaac.corley@my.utsa.edu',
    url=url,
    download_url='{}/archive/{}.tar.gz'.format(url, __version__),
    keywords=[
        'pytorch',
        'image-super-resolution',
        'computer-vision',
        'deep-neural-networks',
    ],
    install_requires=install_requires,
    setup_requires=setup_requires,
    tests_require=tests_require,
    classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
  ],
)
