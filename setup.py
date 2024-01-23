import setuptools
import re

with open("README.md", "r") as fh:
  long_description = fh.read()

requires = [
    # 'numpy>=1.20.3',
    # 'pandas>=1.3.4',
    # 'matplotlib>=3.4.3',
    # 'seaborn>=0.10.0',
    'SciencePlots==1.0.9'
]

def get_property(prop, project):
    result = re.search(r'{}\s*=\s*[\'"]([^\'"]*)[\'"]'.format(prop), open(project + '/__init__.py').read())
    return result.group(1)

setuptools.setup(
  name="freeplot",
  version=get_property('__version__', 'freeplot'),
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