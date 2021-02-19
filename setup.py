#!/usr/bin/env/python
from setuptools import setup
import os

setup(name = "TCRpower",
      description = "Power Calculator for T-cell Receptor detection",
      author = "Gabriel Balaban",
      url = 'https://github.com/GabrielBalabanResearch/TCRpower',
      packages = ['tcrpower'],
      scripts = [os.path.join("scripts", script) for script in  os.listdir("scripts")])