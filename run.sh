
#!/bin/sh 
echo Running all 	
 

  #minimum should be 2000 samples with swap of 0.01
 
for x in  1
	do
		for prob in  2 #2 4 6
		do
			
			# python ptBayeslands_sedvec.py -p $prob -s 10000 -r 10 -t 10 -swap 0.01 -b 0.25 -pt 0.5
 			python ptBayeslands_sedvec_adaptive.py -p $prob -s 10000 -r 10 -t 10 -swap 0.1 -b 0.05 -pt 0.5
 
 
		done
	done 



 