import setuptools

package_dependencies = [
    ('numpy', '1.14.0'),
    ('matplotlib', '2.0.0'),
    ('scipy', '1.2.1'),
    ('scikit-learn', '0.19.0'),
    ('tqdm', '4.0.1'),
    ('torch', '1.5.0'),
    ('torchvision', '0.6.0'),
    ('pytorch-lightning', '0.7.6')
]

dependencies = [f'{p}>={v}' for p, v in package_dependencies]

with open("../README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ddlk",
    version="0.0.0.1",
    packages=setuptools.find_packages(),
    author="Mukund Sudarshan",
    author_email="ms7490+pip@nyu.edu",
    description="Deep direct likelihood knockoffs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rajesh-lab/ddlk",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Operating System :: OS Independent", "Topic :: Scientific/Engineering"
    ],
    python_requires='>=3.6',
    install_requires=dependencies)
