from setuptools import setup, find_packages

setup(
    name="ai-project",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "torch",
        "torchvision",
        "gradio"
    ],
    entry_points={
        "console_scripts": [
            "train-model=src.train:main",
            "predict-model=src.predict:main",
        ],
    },
    python_requires=">=3.8",
)

