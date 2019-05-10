
#!/bin/sh 
echo Running all 	
 

  #minimum should be 2000 samples with swap of 0.01
 
for prob  in  1 2 3
	do
		for x in  1 2 3 4
		do
			
			
 			python ptBayeslands_sedvec.py -p $prob -s 10000 -r 10 -t 5 -swap 0.01 -b 0.4 -pt 0.5 -step 0.05
 			#python ptBayeslands_sedvec_adaptive.py -p $prob -s 50000 -r 20 -t 5 -swap 0.005 -b 0.4 -pt 0.5 -step 0.05
 			 
 
 
		done
	done 



 