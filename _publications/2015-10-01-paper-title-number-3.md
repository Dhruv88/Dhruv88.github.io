---
title: "Memory Efficient GPU based CNN training framework"
collection: publications
permalink: /publication/2015-10-01-paper-title-number-3
excerpt: 'This paper is about the number 3. The number 4 is left for future work.'
date: 2015-10-01
---
Developed a CNN training framework using C++ and CUDA. Explored different ways to make the framework memory efficient to enable it to train large CNNs like AlexNet in single GPU with 12GB memory. Experimented different methods to offload CNN layers to disk when its computation is done for an epoch and prefetch them in the next epoch. Used priority queue based offloading to offload the largest layers first to create enough space for next layers and reducing the number of offload operations performed in an epoch.