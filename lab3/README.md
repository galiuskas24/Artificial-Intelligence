## Naive Bayes

Run: `python classifier.py -t 1`

## Reinforcement Learning
### Value iteration
Run: `python gridworld.py -a value -i 5`

### Q-Learning with epsilon-greedy
Run: `python crawler.py`

### Q-Learning (approximation)

<p align="center">
<img src="http://latex.codecogs.com/gif.latex?%24Q%28s%2C%20a%29%3D%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%20f_%7Bi%7D%28s%2C%20a%29%20w_%7Bi%7D%24">
</p>


Run: `python pacman.py -p ApproximateQAgent -a extractor=SimpleExtractor -x 50 -n 60 -l mediumGrid`

Run: `python pacman.py -p ApproximateQAgent -a extractor=SimpleExtractor -x 50 -n 60 -l mediumClassic`
