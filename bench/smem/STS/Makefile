all:
	python -m turingas.main -i sts32.sass -o sts32.cubin
	python -m turingas.main -i sts64.sass -o sts64.cubin
	python -m turingas.main -i sts128.sass -o sts128.cubin
	python -m turingas.main -i sts128_0.sass -o sts128_0.cubin
	python -m turingas.main -i sts64_2bank_conflict.sass -o sts64_2bank_conflict.cubin
	python -m turingas.main -i sts64_broadcast.sass -o sts64_broadcast.cubin
	python -m turingas.main -i sts64_opt3.sass -o sts64_opt3.cubin
	nvcc -arch=sm_75 main.cu -lcuda -o test

clean:
	sts32.cubin sts64.cubin sts128.cubin sts128_0.cubin sts64_2bank_conflict.cubin sts64_broadcast.cubin sts64_opt3.cubin test
