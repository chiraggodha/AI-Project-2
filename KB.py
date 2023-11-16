import math

# A Knowledge Base that will store pair probabilities for bot 8
class KnowledgeBase:
    def __init__(self, probs = {}):
        self.probs = probs

    def add_prob_pair(self,obj1, obj2, p):
        key = frozenset([obj1, obj2])
        self.probs[key] = p

    def get_probability_for_pair(self, obj1, obj2):
        pair = frozenset([obj1, obj2])
        return self.probs.get(pair, 0)

    def total_obj_prob(self, obj):
        p = sum(self.probs[pair] for pair in self.probs.keys() if obj in pair)
        return p


    def set_obj_pairs(self, obj, val):
        for pair in self.probs.keys():
            if obj in pair:
                self.probs[pair] = val

    def set_nonobj_pairs(self, obj, val):
        for pair in self.probs.keys():
            if obj not in pair:
                self.probs[pair] = val
        

    # Generates the initial knowledge base given a ship object
    def genInitKB(self, ship):
        n = len(ship.open)
        opens = list(ship.open)
        num_pairs = math.comb(n, 2)
        init_prob = 1/num_pairs
        
        for a in range(n):
            for b in range(a+1, n):
                if a!=b and (opens[a] not in ship.closed and opens[b] not in ship.closed):
                    self.add_prob_pair(opens[a], opens[b], init_prob)

    def get_max_locations(self):
        max_prob = max(self.probs.values())
        locations = set()
        for pair, prob in self.probs.items():
            if prob == max_prob:
                loc1, loc2 = pair
                locations.add(loc1)
                locations.add(loc2)
        return locations

    # Get sum of all probabilities in the knowledge base and then divide all the current probabilities by this sum
    def normalize(self):
        normalization_factor = 1/(sum(probs for probs in self.probs.values()))
        for pair, prob in self.probs.items():
            self.probs[pair] = normalization_factor*prob


# This code used to be for computing denom in updateProbs8: 
    # if senseRes:
    #     # p(beep) =  sum over all pairs of p(leak in pair)*p(beep i | leaks in pair)
    #     denom = sum(sense8(obj1, obj2, alpha, ship2, dist)[1] * prob for pair, prob in currProbs.probs.items() 
    #                 for obj1, obj2 in [pair])
    # else:
    #     denom = sum((1 - sense8(obj1, obj2, alpha, ship2, dist)[1]) * prob for pair, prob in currProbs.probs.items()
    #                 for obj1, obj2 in [pair])
            

        