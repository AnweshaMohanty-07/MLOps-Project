setup:
	python3 -m venv ../.PSET2
	#source ../.PSET2/bin/activate

install:
	pip install --upgrade pip &&\
	  pip install -r requirements.txt
	dvc pull -r origin

configure:
	make setup
	make install
