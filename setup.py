from setuptools import setup

with open("README.md") as f:
    long_description = f.read()

setup(
    name="sethoscope-heatmap",
    version="1.14.dev0",
    description="Generate high quality heatmaps from coordinate data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords="heatmap coordinate GPS map",
    author="Seth Golub",
    author_email="tmp+17657@sethoscope.net",
    url="http://www.sethoscope.net/heatmap/",
    license="GPLv3",
    py_modules=["heatmap"],
    zip_safe=True,
    python_requires=">=2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, !=3.4.*",
    install_requires=["osmviz", "pillow"],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Multimedia :: Graphics",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    entry_points={"console_scripts": ["heatmap=heatmap:main"]},
)
