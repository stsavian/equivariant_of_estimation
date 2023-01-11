import setuptools

# setuptools.setup(
#     name="FFT_EXPERIMENTS",
#     version="0.1",
#     author="Stefano Savian",
#     author_email="stefano.savian@iit.it",
#     packages=setuptools.find_packages(exclude=("tests", "scripts")),
#     install_requires=[],
#     classifiers=[
#         "Programming Language :: Python :: 3",
#         "License :: OSI Approved :: GPL License",
#         "Operating System :: Linux",
#     ],
    
# )
#https://setuptools.pypa.io/en/latest/userguide/package_discovery.html
setuptools.setup(name="benchmark_networks",
    # ...
    packages=setuptools.find_packages(
        exclude=['additional'],
    ),
    #package_dir={"": "."}
    # ...
)#include=['pkg*'],

