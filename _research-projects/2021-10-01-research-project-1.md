---
title: "Memory Efficient GPU based CNN training framework"
collection: research-projects
permalink: /research-projects/2021-10-01-research-project-1
excerpt: 'Designing a memory efficient CNN training framework to train large CNNs on a single GPU'
dateFrom: 2021-10-01
dateTo: 2022-01-15
mentor: "Dr. Vishwesh Jatala"
---
Developed a CNN training framework using C++ and CUDA. Explored different ways to make the framework memory efficient to enable it to train large CNNs like AlexNet in single GPU with 12GB memory. Experimented different methods to offload CNN layers to disk when its computation is done for an epoch and prefetch them in the next epoch. Used priority queue based offloading to offload the largest layers first to create enough space for next layers and reducing the number of offload operations performed in an epoch.

[Github Link](https://github.com/Dhruv88/hpmoCNN)
