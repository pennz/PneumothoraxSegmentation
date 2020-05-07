export PATH := /home/v/miniconda3/envs/pyt/bin:$(PATH)

PY3=/home/v/miniconda3/envs/pyt/bin/python3
SRC=$(wildcard *.py)

all: $(SRC)
	git push
	cd ../kaggle_runner; $(PY3) -m pytest test_coord.py -k "TestCo" && cd .runners/intercept-resnet-384/ && $(PY3) main.py

clean:
	rm -rf __pycache__ mylogs
submit:
	HTTP_PROXY=$(PROXY_URL) HTTPS_PROXY=$(PROXY_URL) http_proxy=$(PROXY_URL) https_proxy=$(PROXY_URL) kaggle c submit  -f submission.csv -m "Just test(with T)" siim-acr-pneumothorax-segmentation

run_submit:
	python DAF3D/Train.py
	HTTP_PROXY=$(PROXY_URL) HTTPS_PROXY=$(PROXY_URL) http_proxy=$(PROXY_URL) https_proxy=$(PROXY_URL) kaggle c submit  -f submission.csv -m "Just test(with T)" siim-acr-pneumothorax-segmentation

.PHONY: clean
