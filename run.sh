
#!/bin/sh 
echo Running all 	
 

  
 
for x in  1
	do
		for prob in 2
		do
			
			python ptBayeslands_sedvec.py -p $prob -s 3000 -r 10 -t 10 -swap 0.01 -b 0.25 -pt 0.5
 
 
		done
	done 



 