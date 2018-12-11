
#!/bin/sh 
echo Running all 	

# python ptBayeslands_sedvec.py -p 3 -s 600 -r 10 -b 0.05 -pt 0.5

 
for probability in 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1
	do
		for run in 1 2 3
		do
			python ptBayeslands_sedvec.py -p 4 -s 10000 -r 10 -t 10 -swap $probability -b 0.25 -pt 0.5
 
		done
	done 




#python ptBayeslands_sedvec.py -p 6 -s 10000 -r 10 -b 0.10 -pt 0.5

#python ptBayeslands_sedvec.py -p 4 -s 10000 -r 10 -b 0.05 -pt 0.7
#python ptBayeslands_sedvec.py -p 6 -s 10000 -r 10 -b 0.10 -pt 0.7

#python ptBayeslands_sedvec.py -p 5 -s 10000 -r 10 -b 0.15 -pt 0.7
#python ptBayeslands_sedvec.py -p 5 -s 10000 -r 10 -b 0.15 -pt 0.5
