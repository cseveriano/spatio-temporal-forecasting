import numpy as np
import copy
from itertools import combinations
from matlab_matrix_divide import matrix_divide
from scipy.optimize import nnls
from scipy.linalg import qr


def min_u(granules, indices, m):
    """
    Used for convex hull
    :param granules:
    :param indices:
    :param m:
    :return:
    """
    min = np.inf
    index_min = None

    for i in indices:
        for j in range(0, m):
            if granules[i].oGranules[j].u < min:
                min = granules[i].oGranules[j].u
                index_min = i

    return min


def max_U(granules, indices, m):
    """
    Used for convex hull
    :param granules:
    :param indices:
    :param m:
    :return:
    """
    max = -np.inf
    index_max = None

    for i in indices:
        for j in range(0, m):
            if granules[i].oGranules[j].U > max:
                max = granules[i].oGranules[j].U
                index_max = i

    return max


def power(list, power=1):
    """
    Do the power operation over the list's items
    :param list:
    :param power:
    :return:
    """
    return [item ** power for item in list]


def div(list, operand=1):
    """
    Do the division operation over the list' items
    :param list:
    :param operand:
    :return:
    """
    return [item / operand for item in list]


def sqroot(list):
    """
    Do the sqrt operation over the list's items
    :param list:
    :return:
    """
    return [np.sqrt(item) for item in list]


def sub(list1, list2):
    """
    Subtract a list from another item by item
    :param list1:
    :param list2:
    :return:
    """
    if len(list1) != len(list2):
        raise Exception("Listas de tamanho diferente")

    return [list1[i] - list2[i] for i in range(0, len(list1))]


def sum_dot(s, i, p):
    """
    Sum the items of a list multiplied by a factor from p.
    Used for calculation of affine functions
    :param s:
    :param i:
    :param p:
    :return:
    """
    sumd = 0

    for l in range(0, len(i)):
        sumd += s[i[l]] * float(p[l])

    return sumd


def sum_(s):
    """
    Sum items of a list
    :param s:
    :return:
    """
    sumd = 0

    for l in s:
        sumd += l

    return sumd


def sum_specific(s, i):
    """
    Sum the i items of a list
    :param s:
    :param i:
    :return:
    """
    sumd = 0

    for l in i:
        sumd += s[l]

    return sumd


def normalize(array, min, max):
    """
    Normalize data
    :param array:
    :param min:
    :param max:
    :return:
    """
    return [(x - min) / (max - min) for x in array]


def least_sqr(A, B):
    """
    Using least sqaures numpy
    :param A:
    :param B:
    :return:
    """
    return np.linalg.lstsq(np.array(A), np.array(B), rcond=None)[0].tolist()


def lstsqr_pinv(A, B):
    b = copy.deepcopy(B)
    B_crude = copy.deepcopy(b)
    B = []
    for i in b:
        B.append([i])

    list_ret = np.dot(np.linalg.pinv(A), B).tolist()
    list_ = []
    for i in range(0, len(list_ret)):
        for j in range(0, len(list_ret[i])):
            list_.append(float(list_ret[i][j]))

    return list_


def mldivide(A, B):
    A_crude = copy.deepcopy(A)
    A = np.array(A)
    b = copy.deepcopy(B)
    B_crude = copy.deepcopy(b)
    B = []
    for i in b:
        B.append([i])

    B = np.array(B)
    num_vars = A.shape[1]
    rank = np.linalg.matrix_rank(A)
    solutions = []

    if rank == num_vars:
        sol = least_sqr(A_crude, B_crude)
        return sol
    else:
        for nz in combinations(range(num_vars), rank):
            try:
                sol = np.zeros((num_vars, 1))
                sol[nz, :] = np.asarray(np.linalg.solve(A[:, nz], B))
                solutions.append(sol)
            except np.linalg.LinAlgError:
                pass

    dist_to_zero = np.zeros((len(solutions), len(solutions[0])))
    min_dist = np.inf
    closest = []
    solutions = np.array(solutions)

    for i in range(0, len(solutions)):
        aux = 0
        for j in range(0, len(solutions[i].tolist())):
            for k in range(0, len(solutions[i][j].tolist())):
                aux += abs(0 - solutions[i][j][k])
        dist_to_zero[i][j] = aux
        if aux < min_dist:
            min_dist = aux
            closest = [i, j]

    ret_array = []
    array = solutions[closest[0]].tolist()
    for i in array:
        ret_array.append(i[0])

    return ret_array


def mldivide_matlab(A, B):
    """
    Not working
    :param A:
    :param B:
    :return:
    """
    b = copy.deepcopy(B)
    B_crude = copy.deepcopy(b)
    B = []
    for i in b:
        B.append([i])

    c_matrix = matrix_divide.mldivide(A, B).tolist()

    ret_array = []
    for i in c_matrix:
        ret_array.append(i[0].real)

    return ret_array


def solve_minnonzero(A, b):
    A = np.array(A)
    b = copy.deepcopy(b)
    B_crude = copy.deepcopy(b)
    B = []
    for i in b:
        B.append([i])

    x1, res, rnk, s = np.linalg.lstsq(A, B, rcond=-1)
    if rnk == A.shape[1]:
        ret_array = []
        for i in range(0, len(x1)):
            for j in range(0, len(x1[i])):
                ret_array.append(x1[i][j])

        return ret_array  # nothing more to do if A is full-rank

    Q, R, P = qr(A.T, mode='full',pivoting=True)
    Z = Q[:, rnk:].conj()
    C = np.linalg.solve(Z[rnk:], -x1[rnk:])
    array = (x1 + Z.dot(C)).tolist()

    ret_array = []
    for i in range(0, len(array)):
        for j in range(0, len(array[i])):
            ret_array.append(array[i][j])

    return ret_array


def ls_nnls(A, b):
    test_A = np.array(A)

    try:
        test_b = np.array(b)
        output = list(nnls(test_A, test_b)[0])
        return output
    except Exception as e:
        print(str(e))

