from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='auracanaxai',
      version='0.1',
      description='Tree-based XAI approach',
      long_description=readme(),
      keywords='xai ai cart',
      url='https://github.com/detsutut/AraucanaXAI',
      author='Tommaso Buonocore',
      author_email='buonocore.tms@gmail.com',
      license='MIT',
      packages=['auracanaxai'],
      install_requires=[
            'random',
      ],
      include_package_data=True,
      zip_safe=False)