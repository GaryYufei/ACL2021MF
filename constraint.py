import numpy as np
import torch

class cbs_matrix:

    def __init__(self, vocab_size):
        self.matrix = None
        self.vocab_size = vocab_size

    def init_matrix(self, state_size):
        self.matrix = np.zeros((1, state_size, state_size, self.vocab_size), dtype=np.uint8)

    def add_connect(self, from_state, to_state, w_group):
        assert self.matrix is not None
        for w_index in w_group:
            self.matrix[0, from_state, to_state, w_index] = 1
            self.matrix[0, from_state, from_state, w_index] = 0

    def add_connect_except(self, from_state, to_state, w_group):
        excluded_group_word = [w for w in range(self.vocab_size) if w not in w_group]
        self.add_connect(from_state, to_state, excluded_group_word)

    def init_row(self, state_index):
        assert self.matrix is not None
        self.matrix[0, state_index, state_index, :] = 1

    def get_matrix(self):
        return self.matrix

def CBSConstraint(CBS_type, max_constrain_num):
    if CBS_type == 'Two':
        assert max_constrain_num <= 2
        return TwoConstraint()
    elif CBS_type == 'GBS':
        return GBSConstraint(max_constrain_num)
    else:
        raise NotImplementedError

class Constraint:

    constraint_max_length = 6
    _num_cls = {}
    _cache = {}
        
    def connect_edge(self, M, additional_state, from_state, to_state, constraint):
        queue = [(from_state, c) for c in constraint]
        new_queue = []
        index2state = {}
        while len(queue) > 0:
            (f_state, c) = queue.pop(0)
            if len(c) == 1:
                M.add_connect(f_state, to_state, c)
            else:
                if c[0] not in index2state:
                    index2state[c[0]] = additional_state
                    additional_state += 1
                M.add_connect(f_state, index2state[c[0]], [c[0]])
                if not f_state == from_state:
                    M.add_connect_except(f_state, from_state, [c[0]])
                new_queue.append((index2state[c[0]], c[1:]))

            if len(queue) == 0 and len(new_queue) > 0:
                queue = new_queue
                new_queue = []
                index2state = {}

        return M, additional_state

class TwoConstraint(Constraint):

    def __init__(self):
        super(TwoConstraint).__init__()
        self.state_size = 100 #4 * self.constraint_max_length

    def select_state_func(self, beam_prediction, image_ids):
        bp = []
        for i, image_id in enumerate(image_ids):
            if self._num_cls[image_id] == 0:
                bp.append(beam_prediction[i, 0].unsqueeze(0))
            elif self._num_cls[image_id] == 1:
                bp.append(beam_prediction[i, 1].unsqueeze(0))
            elif self._num_cls[image_id] == 2:
                bp.append(beam_prediction[i, 3].unsqueeze(0))
        return torch.cat(bp, dim=0)
                    
    def get_state_matrix(self, output_size, constraints, image_id):
        assert len(constraints) <= 2
        M = cbs_matrix(output_size)
        M.init_matrix(self.state_size)
        
        self._num_cls[image_id] = len(constraints)
        con_str = []
        for c in constraints:
            c_list = ['#'.join([str(i) for i in x]) for x in c]
            con_str.append('^'.join(c_list))
        marker = '*'.join(con_str) if len(con_str) > 0 else '***'
        if marker not in self._cache:
            if self._num_cls[image_id] == 0:
                additional_state = 1
                for i in range(1):
                    M.init_row(i)
            elif self._num_cls[image_id] == 1:
                for i in range(2):
                    M.init_row(i)
                additional_state = 2
                c1 = constraints[0]
                c1 = [w[:self.constraint_max_length + 1] for w in c1]
                M, additional_state = self.connect_edge(M, additional_state, 0, 1, c1)
            else:
                for i in range(4):
                    M.init_row(i)
                additional_state = 4
                c1, c2 = constraints[0], constraints[1]
                c1 = [w[:self.constraint_max_length + 1] for w in c1]
                c2 = [w[:self.constraint_max_length + 1] for w in c2]
                M, additional_state = self.connect_edge(M, additional_state, 0, 1, c1)
                M, additional_state = self.connect_edge(M, additional_state, 0, 2, c2)
                M, additional_state = self.connect_edge(M, additional_state, 1, 3, c2)
                M, additional_state = self.connect_edge(M, additional_state, 2, 3, c1)
            
            self._cache[marker] = (M.get_matrix(), additional_state)
        
        return self._cache[marker]


class GBSConstraint(Constraint):

    def __init__(self, max_constrain_num):
        super(GBSConstraint).__init__()
        self.state_size = 100 #(max_constrain_num ** 2) * (self.constraint_max_length - 1) + max_constrain_num + 1
        self.max_constrain_num = max_constrain_num

    def get_state_matrix(self, output_size, constraints, image_id):
        assert len(constraints) <= self.max_constrain_num

        M = cbs_matrix(output_size)
        M.init_matrix(self.state_size)

        self._num_cls[image_id] = len(constraints)
        con_str = []
        for c in constraints:
            c_list = ['#'.join([str(i) for i in x]) for x in c]
            con_str.append('^'.join(c_list))
        marker = '*'.join(con_str) if len(con_str) > 0 else '***'

        if marker not in self._cache:
            comb_constrains = []
            for c in constraints:
                comb_constrains += c

            additional_state = len(constraints) + 1
            for i in range(additional_state):
                M.init_row(i)

            for i in range(len(constraints)):
                M, additional_state = self.connect_edge(M, additional_state, i, i + 1, comb_constrains)

            self._cache[marker] = (M.get_matrix(), additional_state)
        return self._cache[marker]

    def select_state_func(self, beam_prediction, image_ids):
        bp = []
        for i, image_id in enumerate(image_ids):
            bp.append(beam_prediction[i, self._num_cls[image_id]].unsqueeze(0))
        return torch.cat(bp, dim=0)
