import setuptools

with open("README.md", "r") as fh:
  long_description = fh.read()

requires = [
    'numpy>=1.18.1',
    'pandas>=1.0.1',
    'scipy>=1.4.1',
    'scikit-learn>=0.23.2',
    'matplotlib>=3.1.3',
    'seaborn>=0.10.0',
    'SciencePlots>=1.0.5'
]

setuptools.setup(
  name="freeplot",
  version="0.0.9",
  author="MTandHJ",
  author_email="congxueric@gmail.com",
  description="a Python data visualization library based on matplotlib",
  long_description=long_description,
  long_description_content_type="text/markdown",
  license='MIT License',
  url="https://github.com/MTandHJ/freeplot",
  packages=setuptools.find_packages(),
  python_requires='>=3.6',
  install_requires=requires,
  classifiers=[
  "Programming Language :: Python :: 3",
  "License :: OSI Approved :: MIT License",
  "Operating System :: OS Independent",
  ],
)