.PHONY: clean

env/bin/python:
	virtualenv env -p python3.5
	env/bin/pip install --upgrade pip
	env/bin/pip install wheel
	touch requirements.txt
	env/bin/pip install -r requirements.txt
	env/bin/python -m spacy.de.download all

clean:
	rm -rfv bin develop-eggs dist downloads eggs env parts
	rm -fv .DS_Store .coverage .installed.cfg bootstrap.py
	rm -fv logs/*.txt
	rm -fv *.pickle
	find . -name '*.pyc' -exec rm -fv {} \;
	find . -name '*.pyo' -exec rm -fv {} \;
	find . -depth -name '*.egg-info' -exec rm -rfv {} \;
	find . -depth -name '__pycache__' -exec rm -rfv {} \;
