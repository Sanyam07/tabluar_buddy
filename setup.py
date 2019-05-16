from __future__ import absolute_import
import sys
import os
import setuptools




if __name__ == "__main__":
    CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
    sys.path.insert(0, CURRENT_DIR)

    with open("README.md", "r") as fh:
        long_description = fh.read()

    setuptools.setup(
                     name                          = "kagglebuddy",
                     version                       = "0.0.3",
                     author                        = "鲲(China)",
                     author_email                  = "972775099@qq.com",
                     description                   = "machine learning toolkit",
                     long_description              = long_description,
                     long_description_content_type = "text/markdown",
                     url                           = "https://github.com/NickYi1990/kaggleBuddy",
                     classifiers=[
                                  "Programming Language :: Python :: 3",
                                  "License :: OSI Approved :: MIT License",
                                  "Operating System :: OS Independent",
                                  ],
                     )
