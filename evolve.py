#!/usr/bin/env python
import random
import json
import subprocess

import numpy as np

from deap import algorithms, base, creator, tools
from scoop import futures

def evaluate(individual):
    try:
        return evaluate_inner(individual)
    except:
        return (0.0,)

def evaluate_inner(individual):
    # These scripts weren't designed to be used as a module, so we invoke
    # them as subprocesses.
    detector = subprocess.run([
        'python', 'detector.py',
        '--video', 'data/201708/20170821170158310.avi',
        '--timeout', '0', # gives us individual frame timestamps
        '--interesting', str(individual[0]),
        '--blur-factor', str(individual[1]),
        '--threshold', str(individual[2]),
        '--dilations', str(individual[3]),
        '--bg-weight', str(individual[4]),
    ], capture_output=True, check=True)

    with open('detector-out.json', 'wb') as f:
        f.write(detector.stdout)

    evaluator = subprocess.run([
        'python', 'evaluate.py',
        '--video', 'data/201708/20170821170158310.avi',
        '--clickpoints',
            'data/clickpoints/20170821170158310_finalstartstopandtrack.cdb',
        '--json', '-'
    ], capture_output=True, input=detector.stdout, check=True)

    # Compute a score from the output
    # Output contains some junk in stdout we need to strip out
    output = evaluator.stdout[evaluator.stdout.index(b'{'):]
    result = json.loads(output)
    if result['detection_ranges'] == 0:
        score = 0.0
    else:
        true_pos_score = \
            result['dr_true_positives'] / result['detection_ranges']
        false_neg_score = 1 - \
            (result['false_negatives'] / result['detection_ranges'])
        score = true_pos_score + false_neg_score  # addition OK?
    return (score,)


def main():
    creator.create('FitnessMax', base.Fitness, weights=(1.0,))
    creator.create('Individual', list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # Parallelize
    toolbox.register('map', futures.map)

    # Attribute generators
    def random_blur_radius():
        r = random.randrange(30)
        if r == 0:
            return 0
        return 2*r - 1

    toolbox.register('attr_interesting', random.uniform, 0.0, 1.0)
    toolbox.register('attr_blur_factor', random_blur_radius)
    toolbox.register('attr_threshold', random.randrange, 100)
    toolbox.register('attr_dilations', random.randrange, 20)
    toolbox.register('attr_bg_weight', random.uniform, 0.0, 1.0)

    # Structure initializers
    toolbox.register('individual', tools.initCycle, creator.Individual, (
        toolbox.attr_interesting,
        toolbox.attr_blur_factor,
        toolbox.attr_threshold,
        toolbox.attr_dilations,
        toolbox.attr_bg_weight,
    ), n=1)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)

    toolbox.register('evaluate', evaluate)
    toolbox.register('mate', tools.cxTwoPoint)
    toolbox.register('mutate', tools.mutFlipBit, indpb=0.05)
    toolbox.register('select', tools.selTournament, tournsize=3)

    # Run the evolution
    pop = toolbox.population(n=2)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register('avg', np.mean)
    stats.register('std', np.std)
    stats.register('min', np.min)
    stats.register('max', np.max)
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=3, 
                                   stats=stats, halloffame=hof, verbose=True)

    # Print hall of fame
    print(hof)
    return pop, log, hof

if __name__ == '__main__':
    main()
