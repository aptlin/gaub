# * Libraries


from setuptools import setup


# * Helpers


def readme():
    with open('README.org') as f:
        return f.read()


# * Configuration


setup(
    name='gaub',
    packages=['gaub'],
    version='0.1',
    description='Visualisation of the gaussian method made easy.',
    long_description=readme(),
    author='Sasha Illarionov',
    author_email='sasha.delly@gmail.com',
    license='MIT',
    url='https://github.com/sdll/gaub',
    download_url='https://github.com/sdll/gaub/archive/0.1.tar.gz',
    keywords=['linear algebra', 'gaussian', 'linear equations'],
    include_package_data=True,
    classifiers=['Development Status :: 3 - Alpha',
                 'Intended Audience :: Education',
                 'License :: OSI Approved :: MIT License',
                 'Natural Language :: English',
                 'Programming Language :: Python :: 3.5',
                 'Programming Language :: Python :: 3.6']
)
