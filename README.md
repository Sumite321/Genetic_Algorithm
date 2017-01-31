# Genetic_Algorithm

This is a simple genetic algorithm part of an University Assignment; Artificial Intelligence. It contains the following main characteristics to evolve a population:
<ul>
<li> Selection - implemented with rank selection here, i.e. the individuals with the highest fitness go to the next stage, in the current implementation that simply means it sorts the population according to fitness. This sorts the array with the individuals with the lowest fitness at the top.
<li> EvaluateFitness - takes the parameters of the i-th individual and evaluates the fitness, saves the fitness value in the last column of the numpy array.
<li> Fitness_function - Computes the "fitness" of a set of parameters.
<li> Mutation - mutates a random subset of the population.
<li> Crossover - chooses a random subset of the population, selects the parent pairs and produces an offspring.
<li> Evolve - evolves the genetic algorithm for the specified number of generations.
</ul>
In addition to this, the evolved population is plotted into a 3D Graph. 

<b> Note : You are required the following modules :</b>
<ul>
<li> numpy
<li> matplotlib

