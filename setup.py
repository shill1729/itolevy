from setuptools import setup

setup(name='itolevy',
      version='0.2.92',
      description='Basic SDE solvers and associated Feynman-Kac solvers.',
      url='https://github.com/shill1729/itolevy',
      author='S. Hill',
      author_email='52792611+shill1729@users.noreply.github.com',
      license='MIT',
      packages=['itolevy'],
      install_requires=[
          'np',
          'pandas',
          "matplotlib",
          "scipy"
      ],
      zip_safe=False)
