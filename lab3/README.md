## Naive Bayes

<p align="center">
<img src="http://latex.codecogs.com/gif.latex?%5Cdpi%7B120%7D%20h_%7BM%20A%20P%7D%3D%5Carg%20%5Cmax%20_%7Bh_%7Bi%7D%20%5Cin%20H%7D%20P%5Cleft%28h_%7Bi%7D%20%7C%20f%5Cright%29%3D%5Carg%20%5Cmax%20_%7Bh_%7Bi%7D%20%5Cin%20H%7D%20%5Cfrac%7BP%5Cleft%28f%20%7C%20h_%7Bi%7D%5Cright%29%20P%5Cleft%28h_%7Bi%7D%5Cright%29%7D%7BP%28f%29%7D%3D%5Carg%20%5Cmax%20_%7Bh_%7Bi%7D%20%5Cin%20H%7D%20P%5Cleft%28f%20%7C%20h_%7Bi%7D%5Cright%29%20P%5Cleft%28h_%7Bi%7D%5Cright%29">
</p>

<p align="center">
<img src="http://latex.codecogs.com/gif.latex?%5Cdpi%7B120%7D%20h_%7BM%20A%20P%7D%3D%5Carg%20%5Cmax%20_%7Bh_%7Bi%7D%20%5Cin%20H%7D%20P%5Cleft%28h_%7Bi%7D%5Cright%29%20%5Cprod_%7Bj%7D%20P%5Cleft%28f_%7Bj%7D%20%7C%20h_%7Bi%7D%5Cright%29">
</p>

Smoothing:
<p align="center">
<img src="http://latex.codecogs.com/gif.latex?%5Cdpi%7B120%7D%20%5Chat%7B%5Ctheta%7D_%7Bi%7D%3D%5Cfrac%7Bx_%7Bi%7D&plus;%5Calpha%7D%7BN&plus;%5Calpha%20d%7D">
</p>

NOTE 1:
We are using smoothing because we want to give the small probability for something with 0 probability.
We do that because we have limited train data and we can not assert with certainty that the probability will be 0.

NOTE 2:
In Bayes formula, if we multiply a lot of small probabilities the result could be a very small number 
(so small that we can not store them to double or long double) and we need to use logarithm.
Logarithm is monotone function and probabilities will be properly transformed.

<p align="center">
<img src="http://latex.codecogs.com/gif.latex?%5Cdpi%7B120%7D%20%5Cbegin%7Balign*%7D%20P%28Hi%7CFj...Fk%29%20%26%3D%20P%28Hi%29*P%28Fj*...*Fk%20%7C%20Hi%29%5C%5C%20%26%3D%20P%28Hi%29*P%28Fj%29*...*P%28Fk%29%5C%5C%20%5C%5C%20log%28P%28Hi%7CFj...Fk%29%29%20%26%3D%20log%28P%28Hi%29*P%28Fj*...*Fk%20%7C%20Hi%29%29%20%5C%5C%20%26%3D%20log%28P%28Hi%29%29%20&plus;%20log%28P%28Fj%7CHi%29%29&plus;...&plus;log%28P%28Fk%7CHi%29%29%20%5Cend%7Balign*%7D">
</p>


Run: `python classifier.py -t 1`  
Run: `python classifier.py --train contest_training --test contest_test -s 1 -l 1`  
->run the naive Bayes classifier on thecontest dataset with smoothing=1 and log scale transformation
## Reinforcement Learning
### Value iteration
<p align="center">
<img src="http://latex.codecogs.com/gif.latex?%5Cdpi%7B120%7D%20V_%7B0%7D%28s%29%3D0">
</p>

<p align="center">
<img src="http://latex.codecogs.com/gif.latex?%5Cdpi%7B120%7D%20V_%7Bk&plus;1%7D%28s%29%20%5Cleftarrow%20%5Cmax%20_%7Ba%7D%20%5Csum_%7Bs%5E%7B%5Cprime%7D%7D%20T%5Cleft%28s%2C%20a%2C%20s%5E%7B%5Cprime%7D%5Cright%29%5Cleft%5BR%5Cleft%28s%2C%20a%2C%20s%5E%7B%5Cprime%7D%5Cright%29&plus;%5Cgamma%20V_%7Bk%7D%5Cleft%28s%5E%7B%5Cprime%7D%5Cright%29%5Cright%5D">
</p>


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

x-> number of train games   
n-> train games + test games