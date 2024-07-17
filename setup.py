import importlib.util
import os
import subprocess
from distutils import cmd as distutils_cmd
from distutils import log as distutils_log
from itertools import chain

import setuptools

spec = importlib.util.spec_from_file_location("package_info", "astravani/package_info.py")
package_info = importlib.util.module_from_spec(spec)
spec.loader.exec_module(package_info)

__contact_emails__ = package_info.__contact_emails__
__contact_names__ = package_info.__contact_names__
__description__ = package_info.__description__
__keywords__ = package_info.__keywords__
__license__ = package_info.__license__
__package_name__ = package_info.__package_name__
__repository_url__ = package_info.__repository_url__
__version__ = package_info.__version__


if os.path.exists("README.md"):
    with open("README.md", "r", encoding="utf-8") as f:
        long_description = f.read()
    long_description_content_type = "text/markdown"

setuptools.setup(
    name=__package_name__,
    version=__version__,
    description=__description__,
    long_description=long_description,
    long_description_content_type=long_description_content_type,
    url=__repository_url__,
    author=", ".join(__contact_names__),  # type: ignore
    author_email=", ".join(__contact_emails__),  # type: ignore
    license=__license__,
    maintainer=", ".join(__contact_names__),  # type: ignore
    maintainer_email=", ".join(__contact_emails__),  # type: ignore
    classifiers=[
        #  1 - Planning
        #  2 - Pre-Alpha
        #  3 - Alpha
        #  4 - Beta
        #  5 - Production/Stable
        #  6 - Mature
        #  7 - Inactive
        "Development Status :: 3 - Alpha",
        # Who is this toolkit for?
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        # Project Domain
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Speech Synthesis",
        "Topic :: Scientific/Engineering :: Speech Processing",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Text Processing :: Linguistic",
        # License
        "License :: OSI Approved :: Apache Software License",
        # Supported Python versions
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        # Additional Settings
        "Environment :: Console",
        "Natural Language :: English",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.11",
    include_package_data=True,
    exclude=["tools", "tests"],
    package_data={"": ["*.txt", "*.md", "*.rst"]},
    zip_safe=False,
    keywords=__keywords__,
)