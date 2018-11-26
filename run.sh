
#!/bin/sh 
echo Running all 	

python ptBayeslands_sedvec.py -p 4 -s 10000 -r 10 -b 0.05 -pt 0.5
python ptBayeslands_sedvec.py -p 6 -s 10000 -r 10 -b 0.10 -pt 0.5
python ptBayeslands_sedvec.py -p 5 -s 10000 -r 10 -b 0.15 -pt 0.5

python ptBayeslands_sedvec.py -p 4 -s 10000 -r 10 -b 0.05 -pt 0.7
python ptBayeslands_sedvec.py -p 6 -s 10000 -r 10 -b 0.10 -pt 0.7
python ptBayeslands_sedvec.py -p 5 -s 10000 -r 10 -b 0.15 -pt 0.7