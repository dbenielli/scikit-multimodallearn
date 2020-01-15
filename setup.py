
import os
from setuptools import setup, find_packages

import multiconfusion


def setup_package():
    """Setup function"""

    name = 'scikit-multimodallearn'
    version = multiconfusion.__version__
    description = 'A scikit-learn compatible package for multimodal Classifiers'
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, 'README.rst'), encoding='utf-8') as readme:
        long_description = readme.read()
    group = 'dev'
    url = 'https://gitlab.lis-lab.fr/{}/{}'.format(group, name)
    project_urls = {
        'Documentation': 'http://{}.pages.lis-lab.fr/{}'.format(group, name),
        'Source': url,
        'Tracker': '{}/issues'.format(url)}
    author = 'Dominique Benielli'
    author_email = 'contact.dev@lis-lab.fr'
    license = 'newBSD'
    classifiers = [
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU Lesser General Public License'
        ' v3 or later (LGPLv3+)',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS'],
    keywords = ('machine learning, supervised learning, classification, '
                'ensemble methods, boosting, kernel')
    packages = find_packages(exclude=['*.tests'])
    install_requires = ['scikit-learn>=0.19', 'numpy', 'scipy', 'cvxopt' ]
    python_requires = '>=3.5'
    extras_require = {
        'dev': ['pytest', 'pytest-cov'],
        'doc': ['sphinx', 'numpydoc', 'sphinx_gallery', 'matplotlib']}
    include_package_data = True

    setup(name=name,
          version=version,
          description=description,
          long_description=long_description,
          url=url,
          project_urls=project_urls,
          author=author,
          author_email=author_email,
          license=license,
          classifiers=classifiers,
          keywords=keywords,
          packages=packages,
          install_requires=install_requires,
          python_requires=python_requires,
          extras_require=extras_require,
          include_package_data=include_package_data)


if __name__ == "__main__":
    setup_package()
