# ClassPipeliner

![Page views](https://visitor-badge.glitch.me/badge?page_id=gmrandazzo.ClassPipeliner)

Binary Classification Pipeliner

## Requirements

- tensorflow
- openbabel
- scikit-learn
- numpy
- codecarbon

## What you will find

Here you will find any binary classification model. 
Just navigate to see the results.

## How to read the data

You have multiple scores.
However, for classification, no matter what, the best score that you can have a look at is the area precision-recall curve.
These values go from 0 to 1. 
Values < 0.5 are not to be considered. The classification models are terrible!
Values > from 0.5, you use your knowledge and responsibility, and you take your decision.

[Here is the knowledge](https://www.datascienceblog.net/post/machine-learning/interpreting-roc-curves-auc/)
[Here is the knowledge](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432)

## Materials and Methods

All the calculations are performed on a Workstation with:

GPU: NVIDIA GeForce RTX 3070
CPU: AMD Ryzen 5 3600 6-Core Processor
RAM: 32GB

and running:

- python 3.7.12
- tensorflow 2.7.0
- scikit-learn 0.24.1
- libscientific 1.4.1

## License

Copyright (C) <2022>  Giuseppe Marco Randazzo <gmrandazzo@gmail.com>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

