COMMIT_MSG = ${m}
GITHUB_BRANCH = Leo


all:
	pip install ./pytorch_extension

run:
	python3 main.py

test:
	make
	make run

uninst:
	pip uninstall sparse_mm -y

clean:
	rm -rf ./pytorch_extension/build
	rm -rf ./pytorch_extension/sparse_mm.egg-info

push:
	git add .
	git commit -m "${COMMIT_MSG}"
	git push origin ${GITHUB_BRANCH}