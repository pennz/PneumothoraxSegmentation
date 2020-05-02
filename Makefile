export PATH := /home/v/miniconda3/envs/pyt/bin:$(PATH)

PY3=/home/v/miniconda3/envs/pyt/bin/python3
SRC=$(wildcard *.py)

all: $(SRC)
	#git push
	echo $(PATH)
	cd ../kaggle_runner; $(PY3) -m pytest test_coord.py -k "TestCo"
	cd .runners/intercept-resnet-384/; $(PY3) main.py

clean:
	rm -rf __pycache__ mylogs

.PHONY: clean
