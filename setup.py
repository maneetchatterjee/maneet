from setuptools import setup, find_packages

setup(
    name="vla_pipeline",
    version="0.1.0",
    description="Vision-Language-Action Pipeline for Robotic Manipulation",
    author="Maneet",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pybullet>=3.2.5",
        "opencv-python-headless>=4.5.0",
        "pillow>=9.0.0",
        "scipy>=1.7.0",
        "matplotlib>=3.5.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ]
    },
)
