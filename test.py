
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from scipy.stats import *
import random
for i in range(10):
    a=expon.cdf(i,1)
    b= 1 if random.random()<a else 0
    print("a: ",a," b:",b)
