# Primer on the Epoch of Reionization

In this series of notebooks I attempt to show the general *mathematical and computational* idea behind simulating the EoR using 
excursion set formalism. This approach is computationally much cheaper compared to the radiation transfer simulations. 
The excursion set formalism was first proposed to be used in this context by 
[Fulanetto et al. (2004)](https://ui.adsabs.harvard.edu/#abs/2004ApJ...613....1F/abstract). 
Since then it was refined and modified in different ways. Probably the most popular implementation of this algorithm is 
[21cmFAST](https://ui.adsabs.harvard.edu/#abs/2011MNRAS.411..955M/abstract).

If you need a general introduction to the physics of reionization, you can start with 
[Pritchard & Loeb (2012)](https://ui.adsabs.harvard.edu/#abs/2012RPPh...75h6901P/abstract). 

In this notebook we consider the simplest regime, when:
* Spin temperature is completely coupled to the kinetic temperature.
* Gas is heated homogeneously.

I plan to make additional notebooks on comic dawn (when the spin temperature couples with gas temperature) later.

___


## Table of contents

The tutorial consists of three notebooks.

1. [The first notebook](01_Preparation.ipynb) is completely devoted to generating the density fields and halo catalogs that 
we will use later.
Also, we look at the functions that allow to smooth the scalar fields with different filters and to calculate the power spectrum. 
This part of the tutorial is not directly related to the reionization, but we will use its results extensively in the next notebooks.


2. In [the second notebook](02_Painting_the_EOR.ipynb) we start to post-process the density field in order to mimic the ionization 
fronts.
We consider two cases:
   * The simplest case, where only the ICs (linear density field) are used.
   * More advanced case with halos.
 

3. In [the third part of the tutorial](03_Filaments.ipynb) we explore a rarely used correction due to semi-neutral filaments inside 
ionizaed regions. 
This effect was studied in [Kaurov & Gnedin (2016)](https://ui.adsabs.harvard.edu/#abs/2016ApJ...824..114K/abstract) and 
[Kaurov 2016](https://ui.adsabs.harvard.edu/#abs/2016ApJ...831..198K/abstract).
Again we consider two cases:
   * The contribution from the semi-neutral filaments to the 21cm power spectrum assuming 
 uniform ionization background inside the bubbles. 
   * Correction to the proximity of halos -- same as above but in the close proximity of the ionizing sources. 
 Some of the ideas are not published.


4. In the fourth notebook we start to explore one of the methods on how to break spherical symmetry in the model. By drawing
[the bubbles in Lagrangian space](04_Lagrangian_bubbles.ipynb) insted of Eulerian space 
one can account for the large scale velocities in the IGM by considering bubbles in the Lagrangian space.


5. The fifth notebook focuses on the [stochastic component](05_Stochasticity.ipynb) which originates in the asymmetry of 
ionization escape fraction and the bursty star formation.
The importance of this effect is studied in [Kaurov (2017)](https://ui.adsabs.harvard.edu/#abs/2017arXiv170904353K/abstract).

6. [In this notebook](06_All_in_one.ipynb) we put everything into [a module](reionprimer.py) and explore the model that we built in the previous sections.

