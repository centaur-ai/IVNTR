"""Setup script."""
from setuptools import find_packages, setup

setup(
    name="predicators",
    version="0.1.0",
    packages=find_packages(include=["predicators", "predicators.*"]),
    install_requires=[
        "open3d==0.18.0",
        "protobuf==3.20.2",
        "z3-solver==4.13.3",
        "numpy==1.23.5",
        "bosdyn-client==4.1.0",
        "pytest==7.1.3",
        "mypy==1.8.0",
        "gym==0.23.1",
        "matplotlib",
        "imageio==2.22.2",
        "imageio-ffmpeg",
        "pandas",
        "wandb",
        "scipy",
        "networkx",
        "tabulate==0.9.0",
        "dill==0.3.8",
        "pyperplan",
        "pathos",
        "requests",
        "slack_bolt",
        "graphlib-backport",
        "openai==1.19.0",
        "pyyaml==6.0.2",
        "pylint==2.14.5",
        "types-PyYAML",
        "lisdf",
        "seaborn==0.12.1",
        "ultralytics",
        "smepy@git+https://github.com/sebdumancic/structure_mapping.git",
        "pg3@git+https://github.com/tomsilver/pg3.git",
        "gym_sokoban@git+https://github.com/Learning-and-Intelligent-Systems/gym-sokoban.git",  # pylint: disable=line-too-long
        "ImageHash",
        "google-generativeai",
        "tenacity",
    ],
    include_package_data=True,
    extras_require={
        "develop": [
            "pytest-cov==2.12.1",
            "pytest-pylint==0.18.0",
            "yapf==0.32.0",
            "docformatter==1.4",
            "isort==5.10.1",
        ]
    })
