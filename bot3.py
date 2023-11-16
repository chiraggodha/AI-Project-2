from ship import ship
import random
from math import exp
# Now I want to work on bot 3
# Bot 3 and leak have random location on open cells
# If bot takes a "sense" action it has p = e^(-a(d-1)) of getting a beep, given leak is d steps away

# d(i, j) is shortest path to (i, j), all we need to do is SSSP with BFS
def SSSP(ship2: ship, src: tuple):
    dist = {cell :  [float('inf'), []] for cell in ship2.open}
    dist[src] = [0,[]]
    fringe = [src]
    while fringe:
        curr = fringe.pop(0)
        currDist = dist[curr][0]

        for neighbor in ship.getOpenNeighbors(ship2, curr):
            new_dist = 1 + currDist
            if new_dist < dist[neighbor][0]:
                dist[neighbor] = [new_dist, dist[curr][1]+[neighbor]]
                fringe.append(neighbor)
    return dist


# Initially all open cells should have the same probability of containing a leak
# What is P(leak is nearby | beep) = P(beep | leak)*P(leak) / P(beep)

def sense(botloc, leakloc, alpha, ship2): #Assuming botloc and leakloc are tuples (row, col)
    dist = SSSP(ship2, botloc)
    d = dist[leakloc][0]
    prob = exp(-alpha*(d-1))
    if random.random() < prob:
        return True, prob
    return False, prob

# To determine the next cell we visit:
# - Determine Max P(leak in cell x)
# Put all cells with this probability into a list
# Go to the closest one from this list

def updateProbs(currProbs, senseRes, ship, dist, alpha):
    newProbs = {}
    if senseRes:
        # Find P(beep)
        denom = sum(currProbs[k] * exp(-alpha*(dist[k][0]-1)) for k in ship.open)
    else:
        # Find P(!beep)
        denom = sum(currProbs[k] * (1 - exp(-alpha*(dist[k][0]-1))) for k in ship.open)
    for cell in ship.open:
        if senseRes:
            p_data_evidence = exp(-alpha * (dist[cell][0]-1))
        else:
            p_data_evidence = 1-exp(-alpha * (dist[cell][0]-1))
        # Essentially, if we get a beep: P(leak in cell)* P(beep at i | leak in cell)/ P(beep)
        new_prob = (currProbs[cell]*p_data_evidence)/denom 
        newProbs[cell] = new_prob
    return newProbs

# Assuming that we went to this cell and we did not find leak
def newCell(cell_loc, currProbs):
    prob_cell = currProbs[cell_loc]
    denom = 1 - prob_cell
    newProbs = {}
    for cell in currProbs.keys():
        newProbs[cell] = currProbs[cell]/denom
    newProbs[cell_loc] = 0
    return newProbs

# Find the cell with max probability, breaking ties by finding min distance
def nextCell(probabilities:dict , distances:dict):
    max_prob = max(probabilities.values())
    max_prob_keys = [key for key,value in probabilities.items() if value==max_prob]
    return min(max_prob_keys, key=lambda key: distances[key][0])

# If we move to a cell and dont detect leak, do we update all probs to be old prob/1 - prob of this cell
def Bot3(ship, alpha):
    opens = list(ship.open)
    while True:
        botloc, leakloc = random.sample(opens, 2)
        if botloc != leakloc: break
    print(f'bot started at: {botloc}, leak is at: {leakloc}')
    start_prob = 1/(len(opens)-1) #Probability leak is in any open cell is 1/# of possible cells
    probs = {cell: start_prob for cell in opens if cell != botloc}
    probs[botloc] = 0
    game_over,t, moves = False,0,[botloc]
    while not game_over:
        senseRes,_ = sense(botloc, leakloc, alpha, ship)
        t+=1 # Sense action takes one time step
        distances = SSSP(ship, botloc)
        probs = updateProbs(probs, senseRes, ship, distances,alpha)
        next_cell = nextCell(probs, distances)
        for loc in distances[next_cell][1]:
            moves.append(loc)
            if loc == leakloc:
                game_over = True
                return game_over, t, moves
            # Set the p(leak in this cell) = 0, and update other probs thru normalization. 
            # Note that this won't change cell with max prob b/c ratio of probabilities will remain the same
            probs = newCell(loc, probs)
            t+=1
        if set(moves)==ship.open: break # This means something went wrong and we fucked up
    return game_over, t, ['Bozo']
    
