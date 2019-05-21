## Naive Bayes

Run: `python classifier.py -t 1`

## Reinforcement Learning
### Value iteration
Run: `python gridworld.py -a value -i 5`

### Q-Learning with epsilon-greedy

<p align="center">
<img src="http://latex.codecogs.com/png.latex?%5Cdpi%7B120%7D%20%24Q_%7B0%7D%28s%2C%20a%29%3D0%24">
</p>

<p align="center">
<img src="http://latex.codecogs.com/png.latex?%5Cdpi%7B120%7D%20%24Q_%7Bk&plus;1%7D%28s%2C%20a%29%20%5Cleftarrow%20Q_%7Bk%7D%28s%2C%20a%29&plus;%28%5Calpha%29%5Cleft%5BR%5Cleft%28s%2C%20a%2C%20s%5E%7B%5Cprime%7D%5Cright%29&plus;%5Cgamma%20%5Cmax%20_%7Ba%5E%7B%5Cprime%7D%7D%20Q_%7Bk%7D%5Cleft%28s%5E%7B%5Cprime%7D%2C%20a%5E%7B%5Cprime%7D%5Cright%29-Q_%7Bk%7D%28s%2C%20a%29%5Cright%5D%24">
</p>


Run: `python crawler.py`

### Q-Learning (approximation)

<p align="center">
<img src="http://latex.codecogs.com/png.latex?%5Cdpi%7B150%7D%20%24Q%28s%2C%20a%29%3D%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%20f_%7Bi%7D%28s%2C%20a%29%20w_%7Bi%7D%24">
</p>

<p align="center">
<img src="http://latex.codecogs.com/png.latex?%5Cdpi%7B150%7D%20difference%24%3D%5Cleft%28r&plus;%5Cgamma%20%5Cmax%20_%7Ba%5E%7B%5Cprime%7D%7D%20Q%5Cleft%28s%5E%7B%5Cprime%7D%2C%20a%5E%7B%5Cprime%7D%5Cright%29%5Cright%29-Q%28s%2C%20a%29%24">
</p>

<p align="center">
<img src="http://latex.codecogs.com/png.latex?%5Cdpi%7B150%7D%20%24w_%7Bi%7D%20%5Cleftarrow%20w_%7Bi%7D&plus;%5Calpha%20%5Ccdot%24difference%24%5Ccdot%20f_%7Bi%7D%28s%2C%20a%29%24">
</p>



Run: `python pacman.py -p ApproximateQAgent -a extractor=SimpleExtractor -x 50 -n 60 -l mediumGrid`

Run: `python pacman.py -p ApproximateQAgent -a extractor=SimpleExtractor -x 50 -n 60 -l mediumClassic`
