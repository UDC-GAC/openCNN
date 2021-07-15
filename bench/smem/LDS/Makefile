all:
	python -m turingas.main -i lds32.sass -o lds32.cubin
	python -m turingas.main -i lds64.sass -o lds64.cubin
	python -m turingas.main -i lds128.sass -o lds128.cubin
	python -m turingas.main -i lds64_opt3.sass -o lds64_opt3.cubin
	nvcc -arch=sm_75 main.cu -lcuda -o test

clean:
	lds32.cubin test lds64.cubin lds64_opt3.cubin lds128.cubin
