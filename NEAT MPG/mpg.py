from __future__ import print_function
import os
import neat
import visualize
from CsvReader import inputs, outputs, testInputs, testOutputs

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        genome.fitness = 0
        #error = 0
        count = 0
        for xi, xo in zip(inputs, outputs):
            output = net.activate(xi)
            #error += abs(output[0] - xo[0]) / xo[0] * 100
            #genome.fitness += abs(output[0] - xo[0])
            genome.fitness += (output[0] - xo[0])**2
            count += 1
        #genome.fitness = 100 - (error / count)
        genome.fitness = 1 - genome.fitness / count
        #genome.fitness =  1 / genome.fitness
        #print(genome.fitness)
    #print("test")
        

def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    #p.add_reporter(neat.Checkpointer(5))

    # Run for up to 300 generations.
    winner = p.run(eval_genomes, 7000)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    for xi, xo in zip(inputs, outputs):
        output = winner_net.activate(xi)
        print("input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))
    #TODO testDaten
    print("validation data")
    totalError = 0
    count = 0
    for xi, xo in zip(testInputs, testOutputs):
        output = winner_net.activate(xi)
        totalError += abs(output[0] - xo[0])
        print("input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))
        count += 1
    print(totalError / count)
    node_names = {-1:'Cylinders', -2: 'Displacement', -3:"Horsepower", -4:"Weight", -5:"Acceleration", -6:"Model Year", -7:"USA", -8:"Europe", -9:"Japan", 0:'MPG'}
    visualize.draw_net(config, winner, True, node_names=node_names)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)

    #p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
    p.run(eval_genomes, 10)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    run(config_path)