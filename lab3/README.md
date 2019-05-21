## Naive Bayes

Run: `python classifier.py -t 1`

## Reinforcement Learning
### Value iteration
Run: `python gridworld.py -a value -i 5`

### Q-Learning with epsilon-greedy
Run: `python crawler.py`

### Q-Learning (approximation)

<p align="center">
<img src="https://drive.google.com/uc?export=view&id=1QuUzKRB1iXJFpOhzUaUgcuGGCbA0-JK7">
</p>

<p align="center">
<span class="math display">\[Q(s, a)=\sum_{i=1}^{n} f_{i}(s, a) w_{i}\]</span>
</p>

*w*<sub>*i*</sub> ← *w*<sub>*i*</sub> + *α*⋅ difference ⋅ *f*<sub>*i*</sub>(*s*, *a*)

difference = (*r*+*γ*max<sub>*a*<sup>′</sup></sub>*Q*(*s*<sup>′</sup>,*a*<sup>′</sup>)) − *Q*(*s*, *a*)


Run: `python pacman.py -p ApproximateQAgent -a extractor=SimpleExtractor -x 50 -n 60 -l mediumGrid`

Run: `python pacman.py -p ApproximateQAgent -a extractor=SimpleExtractor -x 50 -n 60 -l mediumClassic`
