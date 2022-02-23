#!/usr/bin/env python3

from setuptools import setup, find_packages


setup(
    name="dts2repl",
    description="dts2repl",
    author="Antmicro",
    entry_points={
        "console_scripts": ["dts2repl = dts2repl.dts2repl:main"],
    },
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "": ["overlay/*.repl", "models.json"],
    },
)
