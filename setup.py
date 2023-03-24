import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

REQUIRED_PACKAGES = [
    'scikit-learn==0.21.3','jieba==0.42.1',"bert4keras==0.5.9", "mlflow==2.2.1"
]

setuptools.setup(
    name="TextMatch",
    version="0.1.0",
    author="MachineLP",
    author_email="machinelp@163.com",
    description="text matching model library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MachineLP/TextMatch",
    download_url='https://github.com/MachineLP/TextMatch/tags',
    packages=setuptools.find_packages(
        exclude=["tests"]),
    python_requires=">=2.7,!=3.0.*,!=3.1.*,!=3.2.*,!=3.3.*,!=3.4.*",  # '>=3.4',  # 3.4.6
    install_requires=REQUIRED_PACKAGES,
    #extras_require={
    #    "cpu": ["tensorflow>=1.4.0,!=1.7.*,!=1.8.*"],
    #    "gpu": ["tensorflow-gpu>=1.4.0,!=1.7.*,!=1.8.*"],
    #},
    entry_points={
    },
    classifiers=(
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ),
    license="Apache-2.0",
    keywords=['match', 'matching', 'keras'],
)