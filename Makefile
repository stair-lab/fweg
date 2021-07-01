
PURPLE_COLOR=\033[1;34m
NO_COLOR=\033[0m

all:
	@echo "${PURPLE_COLOR}Run make py"

.PHONY: py
py:
	ipython nbconvert src/ipynb/adult/*.ipynb --to script --output-dir src/py/adult
	python clean_gen_pys.py src/py/adult
	@echo "${PURPLE_COLOR}Finished processing adult${NO_COLOR}"

	ipython nbconvert src/ipynb/adult_bb/*.ipynb --to script --output-dir src/py/adult_bb
	python clean_gen_pys.py src/py/adult_bb
	@echo "${PURPLE_COLOR}Finished processing adult_bb${NO_COLOR}"

	ipython nbconvert src/ipynb/adult_periodic/*.ipynb --to script --output-dir src/py/adult_periodic
	python clean_gen_pys.py src/py/adult_periodic
	@echo "${PURPLE_COLOR}Finished processing adult_periodic${NO_COLOR}"

	ipython nbconvert src/ipynb/cifar10/*.ipynb --to script --output-dir src/py/cifar10
	python clean_gen_pys.py src/py/cifar10
	@echo "${PURPLE_COLOR}Finished processing cifar10${NO_COLOR}"

	ipython nbconvert src/ipynb/adience/*.ipynb --to script --output-dir src/py/adience
	python clean_gen_pys.py src/py/adience
	@echo "${PURPLE_COLOR}Finished processing adience${NO_COLOR}"

	ipython nbconvert src/ipynb/adience_ablation/*.ipynb --to script --output-dir src/py/adience_ablation
	python clean_gen_pys.py src/py/adience_ablation
	@echo "${PURPLE_COLOR}Finished processing adience_ablation${NO_COLOR}"
