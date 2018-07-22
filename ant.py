import numpy as np
from collections import namedtuple
from numpy.random import uniform, randint, rand

MAX_CITIES = 30
MAX_DISTANCE = 100
MAX_TOUR = MAX_CITIES * MAX_DISTANCE

ALPHA = 1.0
BETA = 5.0
RHO = 0.5
Q = 100
MAX_TOURS = 20
MAX_TIME = MAX_TOURS * MAX_CITIES
INIT_PHEROMONE = 1.0 / MAX_CITIES
MAX_ANTS = 30

City = namedtuple('City', ['x', 'y'])
best = 1000
best_index = 0


class Ant:
    def __init__(self):
        self.curr_city = randint(MAX_CITIES)
        self.next_city = -1
        self.tabu = np.zeros(MAX_CITIES, dtype=int)
        self.path_index = 1
        self.path = np.ones(MAX_CITIES, dtype=int) * -1
        self.path[0] = self.curr_city
        self.tour_length = 0.0
        self.tabu[self.curr_city] = 1


distance = np.zeros(shape=(MAX_CITIES, MAX_CITIES), dtype=float)
pheromone = np.ones(shape=(MAX_CITIES, MAX_CITIES), dtype=float) * INIT_PHEROMONE

cities = []
ants = []


def init():
    for from_ in range(MAX_CITIES):
        cities.append(City(uniform(0, MAX_DISTANCE), uniform(0, MAX_DISTANCE)))

    for from_ in range(MAX_CITIES):
        for to_ in range(MAX_CITIES):
            if to_ != from_ and distance[from_, to_] == 0.0:
                xd = cities[from_].x - cities[to_].x
                yd = cities[from_].y - cities[to_].y
                distance[from_, to_] = np.sqrt(xd * xd + yd * yd)
                distance[to_, from_] = distance[from_, to_]

    for i in range(MAX_ANTS):
        ants.append(Ant())


def restart_ants():
    global best
    global best_index
    global ants
    ants = []
    for i in randint(MAX_ANTS):
        if ants[i].tour_length < best:
            best = ants[i].tour_length
            best_index = i
        ants.append(Ant())


def ant_product(from_, to_):
    return pheromone[from_, to_]**ALPHA * (1.0/distance[from_, to_]) ** BETA


def select_next_city(i):
    denom = 0.0

    from_ = ants[i].curr_city
    for to_ in range(MAX_CITIES):
        if ants[i].tabu[to_] == 0:
            denom += ant_product(from_, to_)

    assert denom != 0.0

    to_ = 0
    while True:
        if to_ >= MAX_CITIES:
            to_ = 0

        if ants[i].tabu[to_] == 0:
            p = ant_product(from_, to_) / denom
            if rand() < p:
                break
        to_ += 1

    return to_
