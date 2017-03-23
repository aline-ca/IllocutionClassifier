# IllocutionClassifier
Multi-label classification for illocutionary acts via scikit-learn.

# Notes
The makefile creates a virtual python environment (python version 3.5) via virtualenv that is independent on the globally installed python version(s).
It also automatically installs all required python packages for the project (listed in requirements.txt) for this virtual environment independently.

# Installation

1. Install virtualenv globally:
[sudo] pip install virtualenv  

2. Execute makefile:
Write 'make' in the project's root folder.

The subfolder env/bin/ that includes a seperate python and pip version and all necessary packages for the classifier should have been created now.

Now the local pip and python can be executed by writing 'env/bin/pip' or 'env/bin/python'.

3. If you want to delete the virtual environment again:
Write 'make clean' in the project's root folder.
