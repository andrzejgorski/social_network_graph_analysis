import random


def independent_cascade(g, boss, mean_influence=0.2):
    recently_activated = {boss}
    active = {boss}

    g.es["weight"] = 1.0
    for e in g.es:
        g[e.tuple[0], e.tuple[1]] = random.uniform(0, 1)

    while recently_activated:
        new_iteration = set()
        for v in recently_activated:
            not_active_neighbors = set(g.neighbors(v)) - active
            for neighbor in not_active_neighbors:
                if g[v, neighbor] < mean_influence:
                    new_iteration.add(neighbor)
                    active.add(neighbor)
        recently_activated = new_iteration

    return active


def linear_threshold(g, boss):
    recently_activated = {boss}
    active = {boss}
    inactive = set(range(0, g.vcount())) - active

    thresholds = [random.randint(1, g.vcount()) for _ in range(0, g.vcount())]

    while recently_activated:
        new_iteration = set()
        for v in inactive:
            active_neighbors = set(g.neighbors(v)) & active
            if thresholds[v] <= len(active_neighbors):
                new_iteration.add(v)
        active |= new_iteration
        inactive -= new_iteration
        recently_activated = new_iteration

    return active
