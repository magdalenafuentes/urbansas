import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="urbansas",
    version="0.1.0",
    author="Magdalena Fuentes, Bea Steers, Luca Bondi(Robert Bosch GmbH), Julia Wilkins",
    author_email="mfuentes@nyu.edu",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/magdalenafuentes/urbansas",
    download_url='https://github.com/magdalenafuentes/urbansas/releases',
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy==1.20',
        'jupyter',
        'matplotlib',
        'scipy',
        'numba',
        'joblib',
        'scikit-learn',
        'scikit-image',
        'pandas',
        'openpyxl',
        'librosa',
        'moviepy',
        'keras==2.8.0',
        'tensorflow==2.8.0',
        'tensorflow-io==0.23.1',
        'protobuf==3.20.1',
        'opencv-python',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)

