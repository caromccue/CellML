from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

setup(name='cellml',
      version='0.1.1',
      description='Integrated tool to measure the shapes of adherent cells on posts. ',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/caromccue/CellML',
      author='Caroline McCue',
      author_email='caroline.t.mccue@gmail.com',
      license='GPLv3',
      packages=find_packages(exclude=["tests.*", "tests"]),
      install_requires=[
          'click',
          'bleach>=2.1.0',
          'docutils>=0.13.1',
          'Pygments',
          'tensorflow',
          'matplotlib',
          'numpy',
          'opencv-python',
          'pkginfo>=1.4.2',
          'pandas',
          'joblib',
          'pillow>=6.2.0',
          'plotly',
          'scikit-image',
          'setuptools',
          'scipy',
          'seaborn',
          'tensorboard',
          'tqdm',
      ],
      entry_points={
          'console_scripts': [
              'cellml = src.cli:cli',
          ],
      },
      zip_safe=False,
      include_package_data=True)