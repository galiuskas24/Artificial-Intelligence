for i in {5..20}
do
   python runner.py config/rosenbrock_config.json "$i-relu" | cat >> results.txt
done
