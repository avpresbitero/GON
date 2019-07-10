# The Game of Neutrophils: Modeling the Balance Between Apoptosis and Necrosis

contact: avpresbitero@corp.ifmo.ru \
git: https://github.com/avpresbitero/GON

**Abstract**

**Background:** Neutrophils are one of the key players in the human innate immune system (HIIS). In the event of an insult where the body is exposed to inflammation triggering moieties (ITMs), neutrophils are mobilized towards the site of insult and antagonize the inflammation. If the inflammation is cleared, neutrophils go into a programmed death called apoptosis. However, if the insult is intense or persistent, neutrophils take on a violent death pathway called necrosis, which involves the rupture of their cytoplasmic content into the surrounding tissue that causes local tissue damage, thus further aggravating inflammation. This seemingly paradoxical phenomenon fuels the inflammatory process by triggering the recruitment of additional neutrophils to the site of inflammation, aimed to contribute to the complete neutralization of severe inflammation. This delicate balance between the cost and benefit of the neutrophils’ choice of death pathway has been optimized during the evolution of the innate immune system. The goal of our work is to understand how the tradeoff between the cost and benefit of the different death pathways of neutrophils, in response to various levels of insults, has been optimized over evolutionary time by using the concepts of evolutionary game theory.

**Results:** We show that by using evolutionary game theory, we are able to formulate a game that predicts the percentage of necrosis and apoptosis when exposed to various levels of insults.

**Conclusion:** By adopting an evolutionary perspective, we identify the driving mechanisms leading to the delicate balance between apoptosis and necrosis in neutrophils’ cell death in response to different insults. Using our simple model, we verify that indeed, the global cost of remaining ITMs is the driving mechanism that reproduces the percentage of necrosis and apoptosis observed in data and neutrophils need sufficient information of the overall inflammation to be able to pick a death pathway that presumably increases the survival of the organism.

## Prerequisites

1. Python 3.5 or higher
2. Python libraries used:
    - numpy
    - scipy
    - csv
    - matplotlib
    - seaborn
    - pandas
    - itertools
    - os
    - plotly
    - joblib
    - timeit