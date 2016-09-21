# seed
RANDOM=42

for i in {1..100};
do 
	seed=$RANDOM
	echo "python -s $seed -o ../data/ensemble/submission_xgboost_$seed"
done