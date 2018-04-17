# SWATS
This project is about implementing SWATS optimizer as described in this paper https://arxiv.org/pdf/1712.07628.pdf
## What's in this repo
* SWATS.py: Implementation for SWATS as a tensorflow optimizer object
* Pyramid.py: Implementation for the 110-layer PyramidNet as described in this paper https://arxiv.org/abs/1610.02915. Performance of the SWATS optimizer will be evaluated by training the 110-layer PyramidNet on cifar-10 as in the original paper
* progress.py: Simple python helper function to show training progress

## How to use

To use the optimizer in any tensorflow model

```python

from SWATS import SWATS

train_op=SWATS().minimize(loss)
```

To run the test using cifar-10 PyramidNet
```

python Pyramid.py --batch_size --epoch --optimizer
```

To build PyramidNet without training
```python

from Pyramid import Pyramid_110

model=Pyramid_110(alpha)#alpha is the widening factor as described in the paper
graph,inputs,label,is_train,correct_count,train_op,init=pyramid.build(optimizer=optimizer)
```
