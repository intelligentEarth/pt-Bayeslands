
#!/bin/sh 
echo Running all 	
 

  #minimum should be 2000 samples with swap of 0.01
 
for x in  1
	do
		for prob in  1 2 3   
		do
			
			python ptBayeslands_sedvec.py -p $prob -s 50000  -r 20 -t 10 -swap 0.01 -b 0.25 -pt 0.5 
 
 
		done
	done 



 