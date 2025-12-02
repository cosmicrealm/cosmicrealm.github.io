---
title: '小于1000的正整数立方和pair'
date: 2022-05-28
permalink: /posts/2022/05/2022-05-28-blog-algorithm_problem_0/
tags:
  - leetcode
  - algorithm
---


#### 找出所有满足 $a^3+b^3=c^3+d^3$的小于1000的正整数组合

```python
NUM_RANGE = 1000
def find_num1():
    sum_v = {}
    for i in range(1,NUM_RANGE):
        for j in range(1,NUM_RANGE):
            v = i**3 + j**3
            if v not in sum_v.keys():
                sum_v[v] = [[i,j]]
            else:
                sum_v[v].append([i,j])
    res = []
    for pairs in sum_v.values():
        res.extend([p1+p2 for p1 in pairs for p2 in pairs])
    return res
```