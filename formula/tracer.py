
"""
Build a manual tracer for
"""
import itertools


def print_stack(time_step):
    seq = ["x{}".format(i) for i in range(time_step)]


class TraceLSTM(object):
    def __init__(self):
        self.prev_hid = ['h_0']  # we don't need all hidden states, just the previous one
        self.prev_cell = ['c_0']  # we don't need all cell states, just the previous one

        # in the end, since h_0 = 0, c_0 = 0, we filter out terms that contain them!
    def get_init_state(self):
        return ['c_0'], ['h_0']

    def step(self, t, prev_c=None, prev_h=None):
        # add to stack trace one step
        # t: current time step
        if prev_c is None or prev_h is None:
            prev_c, prev_h = self.get_init_state()

        i_t = ['L_sig(W_i x_{})'.format(t)] + apply_non_linear(list_matmul('V_i', prev_h), 'sig') + ['L_sig(b_i)']
        o_t = ['L_sig(W_o x_{})'.format(t)] + apply_non_linear(list_matmul('V_o', prev_h), 'sig') + ['L_sig(b_o)']
        f_t = ['L_sig(W_f x_{})'.format(t)] + apply_non_linear(list_matmul('V_f', prev_h), 'sig') + ['L_sig(b_f)']
        g_t = ['L_tanh(W_g x_{})'.format(t)] + apply_non_linear(list_matmul('V_g', prev_h), 'tanh') + ['L_tanh(b_g)']

        c_t = list_hadmard_mul(f_t, prev_c) + list_hadmard_mul(i_t, g_t)
        h_t = list_hadmard_mul(o_t, apply_non_linear(c_t))

        if t == 1:
            c_t = filter_out_zero(c_t)
            h_t = filter_out_zero(h_t)

        return c_t, h_t

def combine(a):
    return '(' + ' + '.join(a) + ')'


def list_hadmard_mul(a, b):
    combs = itertools.product(a, b)
    # [('a', 'd'), ('a', 'e'), ('b', 'd'), ('b', 'e'), ('c', 'd'), ('c', 'e')]
    return ['(' + ' * '.join(c) + ')' for c in combs]


def list_matmul(a, b):
    return ['(' + a + ' ' + e + ')' for e in b]


def apply_non_linear(l, non_lin='tanh'):
    # apply nonlinear to each element in the list
    return ['L_{}('.format(non_lin) + e + ')' for e in l]


def filter_out_zero(l):
    # return [e for e in l if 'h_0' not in e and 'c_0' not in e]
    return filter(lambda e: 'h_0' not in e and 'c_0' not in e, l)


prev_hid = ['h_0']
prev_cell = ['c_0']
seq = ["x{}".format(i) for i in range(3)]

# o_t = ['L_sig(W_o x_1)', combine(list_matmul(['V_o'], prev_hid)), 'b_o']
# f_t = ['L_sig(W_f x_1)', combine(list_matmul(['V_f'], prev_hid)), 'b_f']
# g_t = ['L_tanh(W_g x_1)', combine(list_matmul(['V_g'], prev_hid)), 'b_g']

i_t = ['L_sig(W_i x_1)'] + list_matmul('V_i', prev_hid) + ['b_i']
o_t = ['L_sig(W_o x_1)'] + list_matmul('V_o', prev_hid) + ['b_o']
f_t = ['L_sig(W_f x_1)'] + list_matmul('V_f', prev_hid) + ['b_f']
g_t = ['L_tanh(W_g x_1)'] + list_matmul('V_g', prev_hid) + ['b_g']

c_t = filter_out_zero(list_hadmard_mul(f_t, prev_cell) + list_hadmard_mul(i_t, g_t))
h_t = filter_out_zero(list_hadmard_mul(o_t, apply_non_linear(c_t)))

# print c_t
# print h_t

prev_hid = h_t
prev_cell = c_t

i_t = ['L_sig(W_i x_1)'] + list_matmul('V_i', prev_hid) + ['b_i']
o_t = ['L_sig(W_o x_1)'] + list_matmul('V_o', prev_hid) + ['b_o']
f_t = ['L_sig(W_f x_1)'] + list_matmul('V_f', prev_hid) + ['b_f']
g_t = ['L_tanh(W_g x_1)'] + list_matmul('V_g', prev_hid) + ['b_g']
# #
# i_t = ['L_sig(W_i x_2)', combine(list_matmul(['V_i'], prev_hid)), 'b_i']
# o_t = ['L_sig(W_o x_2)', combine(list_matmul(['V_o'], prev_hid)), 'b_o']
# f_t = ['L_sig(W_f x_2)', combine(list_matmul(['V_f'], prev_hid)), 'b_f']
# g_t = ['L_tanh(W_g x_2)', combine(list_matmul(['V_g'], prev_hid)), 'b_g']

c_t = filter_out_zero(list_hadmard_mul(f_t, prev_cell) + list_hadmard_mul(i_t, g_t))
h_t = filter_out_zero(list_hadmard_mul(o_t, apply_non_linear(c_t)))

# print len(c_t)
print h_t

# print prev_hid
# print len(i_t)
# print i_t

# tlstm = TraceLSTM()
# c, h = None, None
# for t in range(3):
#     # because t here starts from 0
#     c, h = tlstm.step(t + 1, c, h)
#
# print len(c)
# print
# print len(h)
