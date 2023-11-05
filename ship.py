import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque
from matplotlib.colors import ListedColormap, BoundaryNorm
import heapq


class ship:
    def __init__(self,data, open = set(), fire = set(), button = (), bot = (), closed = set()):
        self.data = data # this should be numpy matrix
        self.open = open # set of tuples, each tuple has indices of open cell
        self.fire = fire # set of tuples, each tuple has indices of fire cell
        self.button = button #tuple
        self.bot = bot # tuple
        self.closed = closed # list of tuples

    @classmethod
    def CreateShip(cls,n):
        ship = cls(np.ones((n,n), dtype = float))
        #Pick random cell in interior to open
        x_rand, y_rand = random.randint(1, ship.data.shape[0]-2) , random.randint(1, ship.data.shape[1]-2)
        ship.data[x_rand][y_rand] = 0
        ship.open = {(x_rand, y_rand)} # We use {} to make a set, now we don't have to worry about any duplicates
        return ship  

    def ShowShip(self): 
        # white = open cells, black = closed cells, red = fire, blue = bot, green = button
        colors = ['white', 'black', 'red', 'blue', 'green']
        custom_map = ListedColormap(colors)
        bounds = [-.1, .5, 1.5, 2.5, 3.5, 4.5]
        norm = BoundaryNorm(bounds, custom_map.N)
        plt.imshow(self.data, cmap = custom_map, norm=norm)
        
        #Ensure shows plot with increments of 1
        plt.xticks(np.arange(-0.5, self.data.shape[1], 1), labels=[])
        plt.yticks(np.arange(-0.5, self.data.shape[0], 1), labels=[])

        #Show gridlines 
        plt.grid(True)

        #Show colorbar
        plt.colorbar()
        plt.show()

    #Helper method that takes in indices and returns list of tuples that is 4 cardinal neighbors
    @staticmethod
    def getNeighbors(ind):
        up = (ind[0]-1, ind[1])
        down = (ind[0]+1, ind[1])
        left = (ind[0], ind[1]-1)
        right = (ind[0], ind[1]+1)
        return [up, down, left, right]
        

    #Helper method for robot motion, returns neighbors that are inbounds, not closed and not on fire
    @staticmethod
    def getAdj(self, ind):
        try:
            up = (ind[0]-1, ind[1])
            if up in self.closed or up[0] >= len(self.data) or up[1] >= len(self.data) or up[0] < 0 or up[1] < 0:  
                up = None
        except IndexError:
            up = None
        try:
            down = (ind[0]+1, ind[1])
            if down in self.closed or down[0] >= len(self.data) or down[1] >= len(self.data) or down[0] < 0 or down[1] < 0:  
                down = None
        except IndexError:
            down = None
        try:
            left = (ind[0], ind[1]-1)
            if left in self.closed  or left[0] >= len(self.data) or left[1] >= len(self.data) or left[0] < 0 or left[1] < 0:  
                left = None
        except IndexError:
            left = None
        try:
            right = (ind[0], ind[1]+1)
            if right in self.closed or right[0] >= len(self.data) or right[1] >= len(self.data) or right[0] < 0 or right[1] < 0:  
                right = None
        except IndexError:
            right = None
        coors = [up, down, left, right]
        for i in coors:
            if i == None:
                coors.remove(i)
        return coors

    # Helper to see if indices are within upper and lower bounds
    @staticmethod
    def checkBounds(upper, lower, ind):
        for i in ind:
            if i > upper or i < lower:
                return False
        return True
    
    #Helper method that takes a list of directions and ensures they're w/in bounds
    def getInbounds(self, dir):
        k = 0
        while k in range(len(dir)):
            if not ship.checkBounds(self.data.shape[0]-1, 0, dir[k]):
                del dir[k]
            else:
                k+=1
        return dir
        

    # Helper method, takes in an open cell and returns list of closed neighbors
    def checkNeighborsOpen(self, openInd):
        neighbors = ship.getNeighbors(openInd)
        valid,k = self.getInbounds(neighbors),0
        while k in range(len(valid)):
            r, c = valid[k] #unpack tuple
            if self.data[r][c] != 1:
                del valid[k]
            else:
                k+=1
        return valid
        
    #Takes in index which is a tuple that represents a cell. Checks to see if it has exactly one open neighbor
    def checkNeighborsClosed(self, Ind):
        neighbors = ship.getNeighbors(Ind)
        valid = self.getInbounds(neighbors)
        numOpen = 0
        for direction in valid:
            i,j = direction
            if self.data[i][j] == 0:
                numOpen+=1
        if numOpen == 1:return True
        return False
    
    #returns the neighboring cells of coordinate that are fire cells
    def checkFireNeighbors(self, coordinate):
        adjCells = self.getNeighbors(coordinate)
        adjCells = self.getInbounds(adjCells)
        fireCells = []
        for cell in adjCells:
            if self.data[cell[0]][cell[1]]==2: fireCells.append((cell[0],cell[1]))
        return fireCells

    #returns a list of open cells with exactly one open neighbor
    def getDeadEnds(self):
        dead_ends = []
        for openInd in self.open:
            if self.checkNeighborsClosed(openInd):
                dead_ends.append(openInd)
        return dead_ends

    def openDeadEnds(self):
        dead_ends = self.getDeadEnds()
        for i in range(len(dead_ends)//2): # Want to open ~half the dead ends
            valid_neighbors = self.getInbounds(ship.getNeighbors(dead_ends[i]))
            closed_neighbors = []
            for neighbor in valid_neighbors:
                i,j = neighbor
                if self.data[i][j] == 1:
                    closed_neighbors.append(neighbor)

            #Now we have closed neighbors of dead end, choose one at random and open
            ri = None
            if len(closed_neighbors)-1 > 0:
                ri = random.randint(0, len(closed_neighbors)-1)
                rx, ry = closed_neighbors[ri]
                self.data[rx][ry] = 0
                self.open.add((rx,ry))

    # In order to more efficiently open the ship we only check the blocked cells that are next to open cells
    # Then we can see which of those blocked cells has exactly 1 open neighbor.
    # While this set is not empty, we choose a random cell in the set to open
    def OpenShip(self):
        while True:         
            possibly_valid_blocked, valid_blocked = [],[]
            for open in self.open:
                possibly_valid_blocked += self.checkNeighborsOpen(open)
            for possibly_valid in possibly_valid_blocked:
                if self.checkNeighborsClosed(possibly_valid): valid_blocked.append(possibly_valid)
            if len(valid_blocked) != 0:
                #choose random valid blocked and open it
                ri = random.randint(0, len(valid_blocked)-1)
                i, j = valid_blocked[ri]
                self.data[i][j] = 0 # Open randomly chosen cell
                self.open.add((i,j)) # Add this to set of open cells
            else:
                break
        self.openDeadEnds()
    

    # Get random spawn points for bot, button and fire
    def chooseInitSpawns(self):
        openlist = list(self.open)
        obj = random.sample(openlist, 3)
        for i in range(len(obj)):
            self.open.discard(obj[i])
            x,y = obj[i]
            self.data[x][y] = i+2
        self.button = obj[2]
        self.bot = obj[1]
        self.fire.add(obj[0])
    

    def spreadFire(self, probability):
        gameOver = False
        fireNeighbors = []
        for fireCoor in self.fire:
            fireNeighbors.append(self.getAdj(fireCoor))
        for coordinates in fireNeighbors:
            coordinates = [coor for coor in coordinates if coor is not None] # eliminating 'None' elements from the coordinates list
            # Why are we calling getInbounds? Doesn't getAdj() already ensure we are inbounds?
            # coordinates = self.getInbounds(coordinates)
            for coordinate in coordinates:
                k = len(self.checkFireNeighbors(coordinate))
                prob = (1-((1-probability)**k))
                randNum = (random.random())
                if randNum < prob:
                    self.fire.add(coordinate)
                    self.data[coordinate[0]][coordinate[1]]=2
                    if (self.bot == coordinate) or (self.button == coordinate):
                        gameOver = True
        return gameOver
    

# Adds cells to self.closed
    def populateClosedCells(self):
        for i in range(len(self.data)):
            for j in range(len(self.data[0])):
                if self.data[i][j]==1:
                    self.closed.add((i,j))
    
    #we might be able to get rid of this function
    def checkBotInFire(self):
        for coor in self.fire:
            if coor == self.bot:
                return True
        return False
    
    # helper function to find open, valid children during a BFS traversal
    def findBFSNeighbors(self, cell):
        closed = self.closed
        up = (cell[0] - 1, cell[1])
        if up in self.closed:
            up = None 
        down = (cell[0] + 1, cell[1])
        if down in self.closed:
            down = None
        right = (cell[0], cell[1]+ 1)
        if right in self.closed:
            right = None
        left = (cell[0], cell[1]-1)
        if left in self.closed:
            left = None
        pot = [up, down, left, right]
        valid = []
        for n in pot:
            if n != None:
                if self.checkBounds(self.data.shape[0]-1, 0, n):
                    valid.append(n)
        return valid

    # Bot 1 uses the logic of a simple BFS
    def bot1(self, q):

        self.chooseInitSpawns()

        game = True
        def findPath():
            q = []
            vis = set()
            parent = {}
            startPos = self.bot
            firePos = self.fire.copy()
            firePos = firePos.pop()
            goal = self.button
            q.append(startPos)
            vis.add(startPos)
            parent[startPos] = None
            while len(q) > 0:
                curr = q.pop(0)
                if curr == goal:
                    path = []
                    while curr is not None:
                        path.append(curr)
                        curr = parent[curr]
                    path.reverse()
                    return path
                neighbors = self.findBFSNeighbors(curr)
                for neighbor in neighbors:
                    if neighbor not in vis:
                        if neighbor != firePos:
                            q.append(neighbor)
                            vis.add(neighbor)
                            parent[neighbor] = curr
            return []
    
        # Runner for bot1 performs the simulation
        t = 0
        path = findPath()
        while(game):
            if len(path) == 0:
                return 0
            move = path.pop(0)
            self.moveBot(move)
            if self.bot == self.button:
                return 1
            self.spreadFire(q)
            if self.data[self.bot[0]][self.bot[1]] == 2 or self.data[self.button[0]][self.button[1]] == 2:
                game = False
            t = t + 1
        return 0

    # Helper method that moves a bot accordingly to its path
    def moveBot(self, newPos):
        oldPos = self.bot
        self.open.add(oldPos)
        self.bot = newPos
        self.data[newPos[0]][newPos[1]] = 3
        self.data[oldPos[0]][oldPos[1]] = 0

    # Bot 2 uses the logic of a simple BFS but is recalculated at each time stamp
    def bot2(self, q):

        self.chooseInitSpawns()

        game = True
        def findPath(sP):
            q = []
            vis = set()
            parent = {}
            startPos = self.bot
            firePos = self.fire.copy()
            firePos = firePos.pop()
            goal = self.button
            q.append(startPos)
            vis.add(startPos)
            parent[startPos] = None
            while len(q) > 0:
                curr = q.pop(0)
                if curr == goal:
                    path = []
                    while curr is not None:
                        path.append(curr)
                        curr = parent[curr]
                    path.reverse()
                    return path
                neighbors = self.findBFSNeighbors(curr)
                for neighbor in neighbors:
                    if neighbor not in vis:
                        if neighbor != firePos:
                            q.append(neighbor)
                            vis.add(neighbor)
                            parent[neighbor] = curr
            return []
    
        # Runner for bot 2 that performs the simulations
        t = 0
        while(game):
            path = findPath(self.bot)
            if len(path) == 0:
                if self.data[self.bot[0]][self.bot[1]] == 2 or self.data[self.button[0]][self.button[1]] == 2:
                    return 1
                else:
                    return 0
            self.moveBot(path.pop(1))
            if self.bot == self.button:
                return 1
            self.spreadFire(q)
            if self.data[self.bot[0]][self.bot[1]] == 2 or self.data[self.button[0]][self.button[1]] == 2:
                game = False
            t = t + 1
        return 0
    
    # Bot 3 uses the logic of a BFS but has the fire probability as an additional input, if no paths are found initally reverts to bot1 and bot2 path calculation
    def bot3(self):
        startPos = self.bot
        q = deque([(startPos, [startPos])])
        vis = set()
        vis.add(startPos)
        while q:
            curNode, path = q.popleft()
            if curNode == self.button:
                return path + [curNode]
            neighbors = self.getAdj(self, curNode)
            neighbors = [n for n in neighbors if n is not None]
            for child in neighbors:
                if child not in vis and child not in self.fire and child not in self.fireAdj:
                    q.append((child, path + [child]))
                    vis.add(child)
        return None
    
    #Spreading fire for bot3
    def spreadFireBot3(self, probability):
        gameOver = False
        fireNeighbors = []
        for fireCoor in self.fire:
            fireNeighbors.append(self.getAdj(self,fireCoor))
        for coordinates in fireNeighbors:
            coordinates = [coor for coor in coordinates if coor is not None] # eliminating 'None' elements from the coordinates list
            coordinates = self.getInbounds(coordinates)
            for coordinate in coordinates:
                k = len(self.checkFireNeighbors(coordinate))
                prob = (1-((1-probability)**k))
                randNum = (random.random())
                if randNum < prob:
                    self.fire.add(coordinate)
                    #self.fireAdj.remove(coordinate) #removing the cell that was before adjacent to the fire since it's now part of the fire
                    nbors = self.getAdj(self,coordinate) #getting the neighbors of the recently added fire cell
                    nbors = [n for n in nbors if n is not None]
                    for nbor in nbors: self.fireAdj.add(nbor) #adding the neighbors to the fireAdj set
                    self.data[coordinate[0]][coordinate[1]]=2
                    if (self.bot == coordinate) or (self.button == coordinate):
                        gameOver = True
        return gameOver

    # Runs bot3 simulation 
    def bot3Runner(self):
        time=0
        gameOver= False
        path = self.bot3()
        self.cleanPath(path)  
        while(True and not gameOver):
            move = path[0]
            self.moveBot(move)
            if self.bot == self.button:
                return
            gameOver = self.spreadFireBot3(1)
            time += 1
            self.ShowShip()
            path = self.bot3()
            if path == None: #if bot3 algorithm doesn't work, try bot1 algorithm
                path = self.bot1()
            if path == None: 
                return
            self.cleanPath(path)
        if gameOver:
            print("Game over after "+str(time)+" turns")


    # Use A* to recalculate the path at each timestep
    # Only move once per timestep though. So perhaps we should only keep track of the first move per calculation
    def bot4(self, probability):
        game = True
        # Manhattan distance to goal
        def getHeuristic(current):
            return abs(current[0] - self.button[0]) + abs(current[1] - self.button[1])
        
        # A* algorithm, returns boolean if path is found
        def calculatePath(start):
            fringe = [(getHeuristic(start), start)] #elts in fringe is the priority = totalDist + heuristic + coordinate
            totalDist = {start:0}
            parent = {start:None}

            while fringe:
                curr = heapq.heappop(fringe)
                current = curr[1]
                if current == self.button:
                    path = [current]
                    while current:
                        if parent[current]:
                            path.append(parent[current])
                        current = parent[current]
                    return True, path[::-1]
                possChildren = self.getNeighbors(current)
                possChildren = self.getInbounds(possChildren)
                for child in possChildren:
                    r,c = child
                    # More filtering for valid children
                    if self.data[r][c] == 4 or self.data[r][c] == 0:
                        tmpDist = totalDist[current] + 1
                        if child not in totalDist or totalDist[child] > tmpDist:
                            totalDist[child] = tmpDist
                            parent[child] = current
                            heapq.heappush(fringe, (tmpDist + getHeuristic(child), child))
            return False, []
        
        # @ Each timestep 2 things are happening:
        # 1) The fire is spreading
        # 2) Bot plans path and moves 1 step
        t = 0
        moves = [self.bot]
        while game:
            possible, path = calculatePath(self.bot)
            print(path)
            if not possible: break

            # Make previous bot position open
            self.data[self.bot[0]][self.bot[1]] = 0 
            self.open.add(self.bot)

            # Bot moves one step
            moves.append(path[1])
            self.bot = path[1] 
            self.data[self.bot[0]][self.bot[1]] = 3

            # Check if we arrived
            if self.bot == self.button:
                return True, moves,t
            
            # Spread the fire and check if it engulfed the bot
            self.spreadFire(probability)
            if self.data[self.bot[0]][self.bot[1]] == 2 or self.data[self.button[0]][self.button[1]] == 2:
                game = False, moves, t
            if t%5 == 0: self.ShowShip()
            t+=1
        self.ShowShip()
        return False, moves,t
    
    # Takes ship object and sets everything that's not blocked to open
    def clearShip(self):
        # Remove Fire
        for fireCoord in self.fire:
            i,j = fireCoord
            self.data[i][j] = 0
            self.open.add(fireCoord)
        self.fire = set()
        # Remove bot
        i,j = self.bot
        self.data[i][j] = 0
        self.open.add(self.bot)
        self.bot = None
        # Remove button
        x,y = self.button
        self.data[x][y] = 0
        self.open.add(self.button)
        self.button = None

# Run simulations for a bot for 100 iterations. Return the success rate. Assumes ship is currently clear
    def run_simulation(self, bot, probability,iter = 100):
        count = 0

        # # Want to suppress output
        # sys.stdout = open(os.devnull, 'w')
        for _ in range(iter):
            self.chooseInitSpawns()
            success, path, time = bot(self,probability)
            if success: count+=1
            self.clearShip()
        # sys.stdout = sys.__stdout__
        return count/iter
    
if __name__ == '__main__':
    ship = ship.CreateShip(10)
    ship.OpenShip()
    ship.ShowShip()

