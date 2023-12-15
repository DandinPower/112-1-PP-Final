all:
	pip install ./pytorch_extension

run:
	python3 main.py

test:
	make
	make run

uninst:
	pip uninstall sparse_mm -y