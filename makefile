.PHONY: all full vdd results figures runtime clean

all: vdd results figures

full: all runtime

test: quick_test

vdd:
	./scripts/run_all_vdd_combinations.sh

results: vdd
	./scripts/run_all_results.sh

figures: results
	./scripts/run_all_figures.sh

runtime:
	./scripts/run_runtime_exp.sh

quick_test:
	./scripts/run_quick_test.sh

clean:
	rm -rf ./data/*
	rm -rf ./outputs/*
