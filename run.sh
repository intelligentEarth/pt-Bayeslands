
#!/bin/sh 
echo Running all 	

# python ptBayeslands_sedvec.py -p 3 -s 600 -r 10 -b 0.05 -pt 0.5

  
 
for x in  1
	do
		for prob in 2
		do
			
			python ptBayeslands_sedvec.py -p $prob -s 1000 -r 10 -t 10 -swap 0.01 -b 0.25 -pt 0.5
 
 
		done
	done 





#python ptBayeslands_sedvec.py -p 6 -s 10000 -r 10 -b 0.10 -pt 0.5

#python ptBayeslands_sedvec.py -p 4 -s 10000 -r 10 -b 0.05 -pt 0.7
#python ptBayeslands_sedvec.py -p 6 -s 10000 -r 10 -b 0.10 -pt 0.7

#python ptBayeslands_sedvec.py -p 5 -s 10000 -r 10 -b 0.15 -pt 0.7
#python ptBayeslands_sedvec.py -p 5 -s 10000 -r 10 -b 0.15 -pt 0.5
