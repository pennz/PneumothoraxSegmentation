export PATH := /home/v/miniconda3/envs/pyt/bin:$(PATH)
export CC_TEST_REPORTER_ID := b98f82b9c31d967a3ed2bf931808cf71747219f2a735c02a3939e830026e7ba8

PY3=/home/v/miniconda3/envs/pyt/bin/python3
SRC=$(wildcard *.py)

all: $(SRC)
	git push
	cc-test-reporter before-build
	-coverage run -m pytest .
	coverage xml
	cc-test-reporter after-build -t coverage.py # --exit-code $TRAVIS_TEST_RESULT
push: $(SRC)
	git push
	cd ../kaggle_runner; $(PY3) -m pytest test_coord.py -k "TestCo" # && cd .runners/intercept-resnet-384/ && $(PY3) main.py
clean:
	-rm -rf __pycache__ mylogs
submit:
	HTTP_PROXY=$(PROXY_URL) HTTPS_PROXY=$(PROXY_URL) http_proxy=$(PROXY_URL) https_proxy=$(PROXY_URL) kaggle c submit  -f submission.csv -m "Just test(with T)" siim-acr-pneumothorax-segmentation

run_submit:
	python DAF3D/Train.py
	HTTP_PROXY=$(PROXY_URL) HTTPS_PROXY=$(PROXY_URL) http_proxy=$(PROXY_URL) https_proxy=$(PROXY_URL) kaggle c submit  -f submission.csv -m "Just test(with T)" siim-acr-pneumothorax-segmentation

.PHONY: clean
