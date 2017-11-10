from core.env import PDM
from core.env import State
from copy import copy, deepcopy

def add_transition_to(state, matrix, x, y, maxX, maxY):
    # Nord
    if y - 1 > -1 and matrix[y-1][x][1] != 1:
        state.add_transition(matrix[y-1][x][0], 'N', 1)
    # Est
    if x + 1 < maxX and matrix[y][x+1][1] != 1:
        state.add_transition(matrix[y][x+1][0], 'E', 1)
    # Ouest
    if x -1 > -1 and matrix[y][x-1][1] != 1:
        state.add_transition(matrix[y][x-1][0], 'O', 1)
    # Sud
    if y + 1 < maxY and matrix[y+1][x][1] != 1:
        state.add_transition(matrix[y+1][x][0], 'S', 1)
    state.add_transition(state.id, 'R', 1)




class PDM_laby(PDM):
    
    def __init__(self, input_file):
        PDM.__init__(self)
        w, h = [int(x) for x in next(input_file).split()]

        self.sizeY = int(w)
        self.sizeX = int (h)
        self.coordinates = [None for _ in range(self.sizeY * self.sizeX)]
        self.matrix = [[(0,0) for i in range(self.sizeX)] for j in range(self.sizeY)]

        
        # Création de la matrice
        state_id = 0
        j = 0
        for line in input_file:
            if(j == self.sizeY):
                break
            for i in range(self.sizeX):
                if line[i] == '*':
                    self.matrix[j][i] = (-1, 1)
                elif line[i] == 'E':
                    self.matrix[j][i] = (state_id, 0)
                    self.current_state_id = state_id
                    state_id += 1    
                elif line[i] == 'S':
                    self.matrix[j][i] = (state_id, 2)
                    self.final_state_id = state_id
                    state_id += 1

                else:
                    self.matrix[j][i] = (state_id, 0)
                    state_id += 1
            j += 1

        # Création des états
        for i in range(self.sizeY):
            for j in range(self.sizeX):
                (state_id, state_type) = self.matrix[i][j]
                
                if state_type  == 1:
                    continue
                reward = 1 if state_type  == 2 else 0
                state = State(state_id, reward)

                if state_type == 2:
                    state.is_terminal = True
                self.coordinates[state_id] = (i,j)
                add_transition_to(state, self.matrix, j, i,
                                  self.sizeX, self.sizeY)

                self.add_state(state)

    def print_states_id(self):
        print_matrice = deepcopy(self.matrix)

        # Murs
        for i in range(self.sizeY):
            for j in range(self.sizeX):
                if self.matrix[i][j][1] == 1:
                    print_matrice[i][j] = "#"
                else:
                    id = self.matrix[i][j][0]
                    print_matrice[i][j] = str(id)

        # Affichage
        for i in range(self.sizeY):
            for j in range(self.sizeX):
                print(print_matrice[i][j], end=' ')
            print("")        
                    
    def print_solution(self, agent):
        self.re_init()
        print_matrice = deepcopy(self.matrix)

        # Murs
        for i in range(self.sizeY):
            for j in range(self.sizeX):
                if self.matrix[i][j][1] == 1:
                    print_matrice[i][j] = "#"
                else:
                    print_matrice[i][j] = " "
        # Chemin

        while self.current_state_id != self.final_state_id:
            i,j = self.coordinates[self.current_state_id]
            print_matrice[i][j] = '.'
            action = agent.decide(self.current_state)
            self.next_sa(action)

        
        # Affichage
        for i in range(self.sizeY):
            for j in range(self.sizeX):
                print(print_matrice[i][j], end=' ')
            print("")        
        

