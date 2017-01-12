#!/usr/bin/env python

# lib.py

import math

import numpy as np
import time
import operator
import functools as ft
import matplotlib.pyplot as plt
def prod(iterable):
    return ft.reduce(operator.mul, iterable, 1)


class Inject(object):
    def __init__(self, params=None):
        # default parameters
        self.params = {'premium_rate': 1.2,
                       'claim_size_lambda': [2.0, 0.5],
                       'density_weights': [0.5, 0.5],
                       'poisson_process_lambda': 1.0,
                       'discounting_rate': 0.1,
                       'penalty_coefficient': 1.5,
                       'debug': False}
        # check parameter argument
        if params is not None:
            try:
                for key in self.params.keys():
                    if key in params:
                        self.params[key] = params[key]
            except AttributeError:
                raise Exception('params, if used, should be a dictionary.')
        # check parameters
        try:
            assert isinstance(params['claim_size_lambda'], list)
        except AssertionError:
            raise Exception('claim_size_lambda should a list of floats serving as intensities.')
        try:
            assert isinstance(params['density_weights'], list)
            assert sum(params['density_weights']) == 1.0
            assert len(params['density_weights']) == len(params['claim_size_lambda'])
            self.params['no_densities'] = len(params['claim_size_lambda'])
        except AssertionError:
            raise Exception('density_weights should be a list of floats summed '
                            'to 1 with the same length of claim_size_lambda')
        # initializing constants
        self.r = None
        self.C_list = []
        self.capital_d_delta_w_list = []
        self.capital_d_delta_1_list = []

        self.commonly_used_constants()

        # initializing
        self.capital_b_1_matrix = np.array([[]])
        self.capital_b_2_matrix = np.array([[[]]])
        self.capital_a_matrix = np.array([[[[]]]])

        self.capital_m_matrix = np.array([[]])
        self.capital_p_matrix = np.array([[]])

        self.capital_e_matrix_w = np.array([[[]]])
        self.capital_e_asteric_matrix_w = np.array([[]])

        self.capital_e_matrix_1 = np.array([[[]]])
        self.capital_e_asteric_matrix_1 = np.array([[]])

        self.v_vector = np.array([])
        self.positive_root_list = np.array([])
        self.selected_positive_root_vector = np.array([])

        self.capital_r_matrix = np.array([[[]]])
        self.capital_o_matrix = np.array([[[]]])

        self.capital_y_vector = np.array([])
        self.capital_x_matrix = np.array([[]])

        self.capital_q_vector = np.array([])
        self.result = None


    def commonly_used_constants(self):
        self.r = (self.params['poisson_process_lambda'] + self.params['discounting_rate']) / \
                 self.params['premium_rate']
        self.C_list = [self.params['poisson_process_lambda'] * self.params['density_weights'][index] *
                       self.params['claim_size_lambda'][index] / self.params['premium_rate'] /
                       (self.params['claim_size_lambda'][index] + self.r)
                       for index in range(self.params['no_densities'])]
        self.capital_d_delta_w_list = [self.params['penalty_coefficient'] * self.C_list[index] /
                                       self.params['claim_size_lambda'][index] ** 2.0
                                       for index in range(self.params['no_densities'])]
        self.capital_d_delta_1_list = [self.C_list[index] / self.params['claim_size_lambda'][index]
                                       for index in range(self.params['no_densities'])]

    def capital_a_and_capital_b(self, N):
        self.capital_b_1_matrix = np.zeros((N, N))
        self.capital_b_1_matrix[0, 0] = sum(self.C_list)
        self.capital_b_2_matrix = np.zeros((N, self.params['no_densities'], N))
        for i in range(1, self.params['no_densities'] + 1):
            self.capital_b_2_matrix[0, i - 1, 0] = self.C_list[i - 1]
        for n in range(2, N + 1):
            for j in range(1, n + 1)[::-1]:
                if self.params['debug']:
                    print(n, j)

                if j == n:
                    self.capital_b_1_matrix[n - 1, j - 1] = self.capital_b_1_matrix[n - 2, n - 2] * \
                                                            self.capital_b_1_matrix[0, 0]
                    for i in range(1, self.params['no_densities'] + 1):
                        self.capital_b_2_matrix[n - 1, i - 1, j - 1] = \
                            self.capital_b_2_matrix[n - 2, i - 1, n - 2] * \
                            self.capital_b_2_matrix[0, i - 1, 0]
                elif j == 1:
                    if self.params['debug']:
                        print(self.capital_b_1_matrix[n - 1, j:n].tolist(), range(j + 1, n + 1))

                    self.capital_b_1_matrix[n - 1, j - 1] = \
                        sum([self.capital_b_1_matrix[n - 2, l - 1] *
                             self.capital_b_2_matrix[0, i - 1, 0] /
                             (self.params['claim_size_lambda'][i - 1] + self.r) **
                             (l + 1.0 - j)
                             for l in range(j, n)
                             for i in range(1, self.params['no_densities'] + 1)]) + \
                        sum([self.capital_b_1_matrix[0, 0] *
                             self.capital_b_2_matrix[n - 2, i - 1, k - 1] /
                             (self.params['claim_size_lambda'][i - 1] + self.r) ** k
                             for k in range(1, n)
                             for i in range(1, self.params['no_densities'] + 1)])

                    for i in range(1, self.params['no_densities'] + 1):
                        self.capital_b_2_matrix[n - 1, i - 1, j - 1] = \
                            sum([self.capital_b_2_matrix[n - 2, i - 1, l - 1] *
                                 self.capital_b_1_matrix[0, 0] /
                                 (self.params['claim_size_lambda'][i - 1] + self.r) ** (l + 1 - j)
                                 for l in range(j, n)]) + \
                            sum([self.capital_b_2_matrix[0, i - 1, 0] *
                                 self.capital_b_1_matrix[n - 2, k - 1] /
                                 (self.params['claim_size_lambda'][i - 1] + self.r) ** k
                                 for k in range(1, n)]) - \
                            sum([self.capital_b_2_matrix[n - 2, i - 1, l - 1] *
                                 self.capital_b_2_matrix[0, k - 1, 0] /
                                 (self.params['claim_size_lambda'][i - 1] - self.params['claim_size_lambda'][k - 1]) **
                                 (l + 1.0 - j)
                                 for l in range(j, n)
                                 for k in range(1, self.params['no_densities'] + 1)
                                 if not k == i]) + \
                            sum([self.capital_b_2_matrix[n - 2, k - 1, l - 1] *
                                 self.capital_b_2_matrix[0, i - 1, 0] /
                                 (self.params['claim_size_lambda'][k - 1] -
                                  self.params['claim_size_lambda'][i - 1]) ** l
                                 for l in range(1, n)
                                 for k in range(1, self.params['no_densities'] + 1)
                                 if not k == i])
                else:
                    if self.params['debug']:
                        print(self.capital_b_1_matrix[n - 1, j:n].tolist(), range(j + 1, n + 1))

                    self.capital_b_1_matrix[n - 1, j - 1] = \
                        sum([self.capital_b_1_matrix[n - 2, l - 1] *
                             self.capital_b_2_matrix[0, i - 1, 0] /
                             (self.params['claim_size_lambda'][i - 1] + self.r) **
                             (l + 1.0 - j)
                             for l in range(j, n)
                             for i in range(1, self.params['no_densities'] + 1)]) + \
                        self.capital_b_1_matrix[n - 2, j - 2] * \
                        self.capital_b_1_matrix[0, 0]

                    for i in range(1, self.params['no_densities'] + 1):
                        self.capital_b_2_matrix[n - 1, i - 1, j - 1] = \
                            sum([self.capital_b_2_matrix[n - 2, i - 1, l - 1] *
                                 self.capital_b_1_matrix[0, 0] /
                                 (self.params['claim_size_lambda'][i - 1] + self.r) ** (l + 1 - j)
                                 for l in range(j, n)]) - \
                            sum([self.capital_b_2_matrix[n - 2, i - 1, l - 1] *
                                 self.capital_b_2_matrix[0, k - 1, 0] /
                                 (self.params['claim_size_lambda'][i - 1] -
                                  self.params['claim_size_lambda'][k - 1]) ** (l + 1.0 - j)
                                 for l in range(j, n)
                                 for k in range(1, self.params['no_densities'] + 1)
                                 if not k == i]) + \
                            self.capital_b_2_matrix[n - 2, i - 1, j - 2] * \
                            self.capital_b_2_matrix[0, i - 1, 0]

        self.capital_a_matrix = np.zeros((N, self.params['no_densities'], N, N))
        if N > 1:
            for i in range(1, self.params['no_densities'] + 1):
                self.capital_a_matrix[1, i - 1, 0, 0] = -sum([self.C_list[l - 1] * self.C_list[i - 1] /
                                                              (self.params['claim_size_lambda'][i - 1] + self.r)
                                                              for l in range(1, self.params['no_densities'] + 1)])
        for n in range(2, N + 1):
            for j in range(1, n)[::-1]:
                if j == (n - 1):
                    for i in range(1, self.params['no_densities'] + 1):
                        self.capital_a_matrix[n - 1, i - 1, n - 2, 0] = \
                            -self.capital_b_2_matrix[n - 2, i - 1, n - 2] * \
                            self.capital_b_1_matrix[0, 0] / \
                            (self.params['claim_size_lambda'][i - 1] +
                             self.r)
                else:
                    for k in range(1, n - j + 1):
                        if k == 1:
                            for i in range(1, self.params['no_densities'] + 1):
                                self.capital_a_matrix[n - 1, i - 1, j - 1, k - 1] = \
                                    sum([self.capital_a_matrix[n - 2, i - 1, j - 1, h - 1] *
                                         self.capital_b_2_matrix[0, l - 1, 0] /
                                         (self.params['claim_size_lambda'][l - 1] + self.r) **
                                         (h + 1.0 - k)
                                         for h in range(k, n - j)
                                         for l in range(1, self.params['no_densities'] + 1)]) - \
                                    sum([self.capital_b_2_matrix[n - 2, i - 1, l - 1] *
                                         self.capital_b_1_matrix[0, 0] /
                                         (self.params['claim_size_lambda'][i - 1] + self.r) **
                                         (l + 1.0 - j)
                                         for l in range(j, n)])
                        else:
                            for i in range(1, self.params['no_densities'] + 1):
                                self.capital_a_matrix[n - 1, i - 1, j - 1, k - 1] = \
                                    sum([self.capital_a_matrix[n - 2, i - 1, j - 1, h - 1] *
                                         self.capital_b_2_matrix[0, l - 1, 0] /
                                         (self.params['claim_size_lambda'][l - 1] + self.r) **
                                         (h + 1.0 - k)
                                         for h in range(k, n - j)
                                         for l in range(1, self.params['no_densities'] + 1)]) + \
                                    self.capital_a_matrix[n - 2, i - 1, j - 1, k - 2] * \
                                    self.capital_b_1_matrix[0, 0]
        if self.params['debug']:
            print('capital_a_and_capital_b: Done!')
#################################################################################
        # text for discounted density
    def hdelta1(self,u,x):
        if u<= x:
            return sum([self.C_list[i-1]*np.exp(-self.r*(x-u))for i in range(1,3) ])
        else:
            return sum([self.C_list[i-1]* np.exp(self.params['claim_size_lambda'][i-1]*(x-u)) for i in range(1,3)])

    def hdelta(self,n,u,x):
        self.capital_a_and_capital_b(n)
        if u< x:
            return sum([self.capital_a_matrix[n-1,i-1,j-1,k-1] * u**(j-1)/np.math.factorial(j-1) *
                        x**(k-1)/np.math.factorial(k-1) *np.exp(-self.params['claim_size_lambda'][i-1] * u - self.r * x)
                        for i in range(1, self.params['no_densities']+1) for j in range(1, n) for k in range(1, n-j+1)])\
                    + sum([self.capital_b_1_matrix[n-1,j-1]*(x-u)**(j-1)/np.math.factorial(j-1)*np.exp(-self.r*(x-u))
                           for j in range(1, n+1)])
        else:
            return sum([self.capital_a_matrix[n-1,i-1,j-1,k-1]*u**(j-1)/np.math.factorial(j-1) *
                        x**(k-1)/np.math.factorial(k-1) *np.exp(-self.params['claim_size_lambda'][i-1]*u-self.r*x )
                        for i in range(1, self.params['no_densities']+1) for j in range(1, n) for k in range(1, n-j+1)])\
                    + sum([self.capital_b_2_matrix[n-1, i-1, j-1] * (u-x)**(j-1)/np.math.factorial(j-1)*
                           np.exp(self.params['claim_size_lambda'][i-1]*(x-u))
                           for i in range(1, self.params['no_densities']+1) for j in range(1, n+1)])



    ##################################################################


    def coef_with_capital_b_2_in_capital_m(self, z, k, j, i):
        return sum([1.0 / self.params['claim_size_lambda'][i - 1] ** (k + 1.0 - l) * (-z) ** (l - j) /
                    np.math.factorial(l - j) * np.exp(self.params['claim_size_lambda'][i - 1] * z)
                    for l in range(j, k + 1)]) - \
               1.0 / self.params['claim_size_lambda'][i - 1] ** (k - j + 1.0)

    def coef_with_capital_a_in_capital_m(self, z, k):
        return 1.0 / self.r ** k - \
               sum([1.0 / self.r ** (k + 1.0 - l) * z ** (l - 1.0) / np.math.factorial(l - 1) * np.exp(-self.r * z)
                    for l in range(1, k + 1)])

    def coef_with_capital_b_2_in_capital_p(self, z, k, j, i):
        return 1.0 / self.params['claim_size_lambda'][i - 1] ** (k - j + 2.0) - \
               sum([1.0 / self.params['claim_size_lambda'][i - 1] ** (k - j + 3.0 - l) * (-z) ** (l - 1.0) /
                    np.math.factorial(l - 1) * np.exp(self.params['claim_size_lambda'][i - 1] * z)
                    for l in range(1, k - j + 3)])

    def coef_with_capital_a_in_capital_p(self, z, k):
        return k / self.r ** (k + 1.0) - \
               sum([k / self.r ** (k + 2.0 - l) * z ** (l - 1.0) / np.math.factorial(l - 1) * np.exp(-self.r * z)
                    for l in range(1, k + 2)])

    def capital_m_and_capital_p(self, N, z):
        self.capital_m_matrix = np.zeros((self.params['no_densities'], N))
        for j in range(1, N + 1):
            if j == N:
                for i in range(1, self.params['no_densities'] + 1):
                    self.capital_m_matrix[i - 1, j - 1] = self.capital_b_2_matrix[j - 1, i - 1, j - 1] * \
                                                          (np.exp(self.params['claim_size_lambda'][i - 1] * z) - 1.0) / \
                                                          self.params['claim_size_lambda'][i - 1]
            else:
                for i in range(1, self.params['no_densities'] + 1):
                    self.capital_m_matrix[i - 1, j - 1] = sum([self.capital_b_2_matrix[N - 1, i - 1, k - 1] *
                                                               self.coef_with_capital_b_2_in_capital_m(z, k, j, i)
                                                               for k in range(j, N + 1)]) + \
                                                          sum([self.capital_a_matrix[N - 1, i - 1, j - 1, k - 1] *
                                                               self.coef_with_capital_a_in_capital_m(z, k)
                                                               for k in range(1, N - j + 1)])
        self.capital_p_matrix = np.zeros((self.params['no_densities'], N))
        for j in range(1, N + 1):
            if j == N:
                for i in range(1, self.params['no_densities'] + 1):
                    self.capital_p_matrix[i - 1, j - 1] = \
                        self.capital_b_2_matrix[N - 1, i - 1, N - 1] * \
                        (1.0 - np.exp(self.params['claim_size_lambda'][i - 1] * z) +
                         self.params['claim_size_lambda'][i - 1] * z *
                         np.exp(self.params['claim_size_lambda'][i - 1] * z)) / \
                        self.params['claim_size_lambda'][i - 1] ** 2.0
            else:
                for i in range(1, self.params['no_densities'] + 1):
                    self.capital_p_matrix[i - 1, j - 1] = \
                        sum([self.capital_b_2_matrix[N - 1, i - 1, k - 1] * (k - j + 1.0) *
                             self.coef_with_capital_b_2_in_capital_p(z, k, j, i)
                             for k in range(j, N + 1)]) + \
                        sum([self.capital_a_matrix[N - 1, i - 1, j - 1, k - 1] *
                             self.coef_with_capital_a_in_capital_p(z, k)
                             for k in range(1, N - j + 1)])
        if self.params['debug']:
            print('capital_m_and_capital_p: Done!')
#####################################################
#text for m summation update by ran 20160512
    def m_summation(self, N, z, u):
        time_recorder = time.time()
        self.capital_a_and_capital_b(N)
        self.capital_m_and_capital_p(N, z)
        self.capital_m_summation = sum(self.capital_m_matrix[i - 1, j - 1] *
                                       u ** (j - 1) /
                                       np.math.factorial(j - 1) *
                                       np.exp(-self.params['claim_size_lambda'][i - 1] * u)
                                       for i in range(1, self.params['no_densities'] + 1)
                                       for j in range(1, N + 1))
        if self.params['debug']:
            print(time.time() - time_recorder)
        return self.capital_m_summation

    def p_summation(self, N, z, u):
        time_recorder = time.time()
        self.capital_a_and_capital_b(N)
        self.capital_m_and_capital_p(N, z)
        self.capital_p_summation = sum(self.capital_p_matrix[i - 1, j - 1] *
                                       u ** (j - 1) /
                                       np.math.factorial(j - 1) *
                                       np.exp(-self.params['claim_size_lambda'][i - 1] * u)
                                       for i in range(1, self.params['no_densities'] + 1)
                                       for j in range(1, N + 1))
        if self.params['debug']:
            print(time.time() - time_recorder)
        return self.capital_p_summation
#######################################################
    def capital_e_and_capital_e_asteric(self, N):
        self.capital_e_matrix_w = np.zeros((self.params['no_densities'], N, N))
        for l in range(1, N):
            for j in range(1, l + 2):
                if j == 1:
                    for i in range(1, self.params['no_densities'] + 1):
                        self.capital_e_matrix_w[i - 1, l - 1, j - 1] = \
                            sum([self.capital_a_matrix[l - 1, i - 1, j - 1, k - 1] *
                                 self.capital_d_delta_w_list[h - 1] /
                                 (self.params['claim_size_lambda'][h - 1] + self.r) ** k
                                 for h in range(1, self.params['no_densities'] + 1)
                                 for k in range(1, l - j + 1)]) + \
                            sum([self.capital_b_1_matrix[l - 1, k - 1] *
                                 self.capital_d_delta_w_list[i - 1] /
                                 (self.params['claim_size_lambda'][i - 1] + self.r) ** k
                                 for k in range(1, l + 1)]) + \
                            sum([self.capital_b_2_matrix[l - 1, h - 1, k - 1] *
                                 self.capital_d_delta_w_list[i - 1] /
                                 (self.params['claim_size_lambda'][h - 1] -
                                  self.params['claim_size_lambda'][i - 1]) ** k
                                 for h in range(1, self.params['no_densities'] + 1)
                                 for k in range(1, l + 1)
                                 if not h == i]) - \
                            sum([self.capital_b_2_matrix[l - 1, i - 1, k - 1] *
                                 self.capital_d_delta_w_list[h - 1] /
                                 (self.params['claim_size_lambda'][i - 1] -
                                  self.params['claim_size_lambda'][h - 1]) **
                                 (k + 1 - j)
                                 for h in range(1, self.params['no_densities'] + 1)
                                 for k in range(j, l + 1)
                                 if not h == i])
                elif j < l + 1:
                    for i in range(1, self.params['no_densities'] + 1):
                        self.capital_e_matrix_w[i - 1, l - 1, j - 1] = \
                            sum([self.capital_a_matrix[l - 1, i - 1, j - 1, k - 1] *
                                 self.capital_d_delta_w_list[h - 1] /
                                 (self.params['claim_size_lambda'][h - 1] + self.r) ** k
                                 for h in range(1, self.params['no_densities'] + 1)
                                 for k in range(1, l - j + 1)]) - \
                            sum([self.capital_b_2_matrix[l - 1, i - 1, k - 1] *
                                 self.capital_d_delta_w_list[h - 1] /
                                 (self.params['claim_size_lambda'][i - 1] -
                                  self.params['claim_size_lambda'][h - 1]) **
                                 (k + 1.0 - j)
                                 for h in range(1, self.params['no_densities'] + 1)
                                 for k in range(j, l + 1)
                                 if not h == i]) + \
                            self.capital_b_2_matrix[l - 1, i - 1, j - 2] * \
                            self.capital_d_delta_w_list[i - 1]
                else:
                    for i in range(1, self.params['no_densities'] + 1):
                        self.capital_e_matrix_w[i - 1, l - 1, j - 1] = \
                            self.capital_b_2_matrix[l - 1, i - 1, l - 1] * \
                            self.capital_d_delta_w_list[i - 1]

        self.capital_e_asteric_matrix_w = np.zeros((self.params['no_densities'], N))
        for j in range(1, N + 1):
            if j == 1:
                for i in range(1, self.params['no_densities'] + 1):
                    self.capital_e_asteric_matrix_w[i - 1, j - 1] = \
                        sum([self.capital_e_matrix_w[i - 1, l - 1, j - 1]
                             for l in range(max(1, j - 1), N)]) + \
                        self.capital_d_delta_w_list[i - 1]
            else:
                for i in range(1, self.params['no_densities'] + 1):
                    self.capital_e_asteric_matrix_w[i - 1, j - 1] = \
                        sum([self.capital_e_matrix_w[i - 1, l - 1, j - 1]
                             for l in range(max(1, j - 1), N)])

        self.capital_e_matrix_1 = np.zeros((self.params['no_densities'], N, N))
        for l in range(1, N):
            for j in range(1, l + 2):
                if j == 1:
                    for i in range(1, self.params['no_densities'] + 1):
                        self.capital_e_matrix_1[i - 1, l - 1, j - 1] = \
                            sum([self.capital_a_matrix[l - 1, i - 1, j - 1, k - 1] *
                                 self.capital_d_delta_1_list[h - 1] /
                                 (self.params['claim_size_lambda'][h - 1] + self.r) ** k
                                 for h in range(1, self.params['no_densities'] + 1)
                                 for k in range(1, l - j + 1)]) + \
                            sum([self.capital_b_1_matrix[l - 1, k - 1] *
                                 self.capital_d_delta_1_list[i - 1] /
                                 (self.params['claim_size_lambda'][i - 1] + self.r) ** k
                                 for k in range(1, l + 1)]) + \
                            sum([self.capital_b_2_matrix[l - 1, h - 1, k - 1] *
                                 self.capital_d_delta_1_list[i - 1] /
                                 (self.params['claim_size_lambda'][h - 1] -
                                  self.params['claim_size_lambda'][i - 1]) ** k
                                 for h in range(1, self.params['no_densities'] + 1)
                                 for k in range(1, l + 1)
                                 if not h == i]) - \
                            sum([self.capital_b_2_matrix[l - 1, i - 1, k - 1] *
                                 self.capital_d_delta_1_list[h - 1] /
                                 (self.params['claim_size_lambda'][i - 1] -
                                  self.params['claim_size_lambda'][h - 1]) **
                                 (k + 1 - j)
                                 for h in range(1, self.params['no_densities'] + 1)
                                 for k in range(j, l + 1)
                                 if not h == i])
                elif j < l + 1:
                    for i in range(1, self.params['no_densities'] + 1):
                        self.capital_e_matrix_1[i - 1, l - 1, j - 1] = \
                            sum([self.capital_a_matrix[l - 1, i - 1, j - 1, k - 1] *
                                 self.capital_d_delta_1_list[h - 1] /
                                 (self.params['claim_size_lambda'][h - 1] + self.r) ** k
                                 for h in range(1, self.params['no_densities'] + 1)
                                 for k in range(1, l - j + 1)]) - \
                            sum([self.capital_b_2_matrix[l - 1, i - 1, k - 1] *
                                 self.capital_d_delta_1_list[h - 1] /
                                 (self.params['claim_size_lambda'][i - 1] -
                                  self.params['claim_size_lambda'][h - 1]) **
                                 (k + 1.0 - j)
                                 for h in range(1, self.params['no_densities'] + 1)
                                 for k in range(j, l + 1)
                                 if not h == i]) + \
                            self.capital_b_2_matrix[l - 1, i - 1, j - 2] * \
                            self.capital_d_delta_1_list[i - 1]
                else:
                    for i in range(1, self.params['no_densities'] + 1):
                        self.capital_e_matrix_1[i - 1, l - 1, j - 1] = \
                            self.capital_b_2_matrix[l - 1, i - 1, l - 1] * \
                            self.capital_d_delta_1_list[i - 1]
        self.capital_e_asteric_matrix_1 = np.zeros((self.params['no_densities'], N))
        for j in range(1, N + 1):
            if j == 1:
                for i in range(1, self.params['no_densities'] + 1):
                    self.capital_e_asteric_matrix_1[i - 1, j - 1] = \
                        sum([self.capital_e_matrix_1[i - 1, l - 1, j - 1]
                             for l in range(max(1, j - 1), N)]) + \
                        self.capital_d_delta_1_list[i - 1]
            else:
                for i in range(1, self.params['no_densities'] + 1):
                    self.capital_e_asteric_matrix_1[i - 1, j - 1] = \
                        sum([self.capital_e_matrix_1[i - 1, l - 1, j - 1]
                             for l in range(max(1, j - 1), N)])

        if self.params['debug']:
            print('capital_e_and_capital_e_asteric: Done!')
###################################################################
  # test for e_asteric_sum
    def e_1_summation(self, N, z, u):
        time_recorder = time.time()
        self.capital_a_and_capital_b(N)
        self.capital_m_and_capital_p(N, z)
        self.capital_e_and_capital_e_asteric(N)
        self.capital_e_1_summation = sum(self.capital_e_asteric_matrix_1[i - 1, j - 1] *
                                       u ** (j - 1) /
                                       np.math.factorial(j - 1) *
                                       np.exp(-self.params['claim_size_lambda'][i - 1] * u)
                                       for i in range(1, self.params['no_densities'] + 1)
                                       for j in range(1, N + 1))
        if self.params['debug']:
            print(time.time() - time_recorder)
        return self.capital_e_1_summation

#####################################################################
    def roots_v(self, N):
        self.v_vector = []
        self.positive_root_list = []
        for k in range(0, N):
            a = np.exp(-1.0j * 2 * k * math.pi / N)
            coeff_list = [a,
                          a * (sum(self.params['claim_size_lambda']) - self.r),
                          a * (prod(self.params['claim_size_lambda']) -
                               self.r * sum(self.params['claim_size_lambda'])) -
                          sum([prod(x) for x in zip(self.params['claim_size_lambda'][::-1], self.C_list)]) +
                          sum([x * y for x in self.params['claim_size_lambda'] for y in self.C_list]) +
                          self.r * sum(self.C_list),
                          -a * prod(self.params['claim_size_lambda']) * self.r +
                          prod(self.params['claim_size_lambda']) * sum(self.C_list) +
                          self.r * sum([prod(x) for x in zip(self.params['claim_size_lambda'][::-1], self.C_list)])]
            root_vec = np.roots(coeff_list)
            if self.params['debug']:
                print(root_vec)
            negative_root_list = list(filter(lambda x: x < 0, root_vec))
            positive_root_list = list(filter(lambda x: x >= 0, root_vec))
            if negative_root_list.__len__() != 2:
                raise ValueError('Two negative roots are expected!')
            self.v_vector = self.v_vector + negative_root_list
            self.positive_root_list = self.positive_root_list + positive_root_list
        self.v_vector = np.array(self.v_vector)
        self.positive_root_list = np.array(self.positive_root_list)
        self.selected_positive_root_vector = np.array(list(filter(lambda x: x < self.r, self.positive_root_list)))
        if self.params['debug']:
            print('roots_v: Done!')

    def capital_r_and_capital_o(self, N, z):
        self.capital_r_matrix = np.zeros((self.params['no_densities'], N, self.params['no_densities'] * N)) * 1.0j
        for j in range(1, N):
            for h in range(1, self.params['no_densities'] * N + 1):
                for i in range(1, self.params['no_densities'] + 1):
                    self.capital_r_matrix[i - 1, j - 1, h - 1] = \
                        sum([self.capital_a_matrix[N - 1, i - 1, j - 1, k - 1] /
                             (self.r - self.v_vector[h - 1])**(k + 1 - l) *
                             z ** (l - 1) /
                             np.math.factorial(l - 1) *
                             np.exp(-self.r * z)
                             for k in range(1, N - j + 1)
                             for l in range(1, k + 1)])
        self.capital_o_matrix = np.zeros((self.params['no_densities'], N, self.params['no_densities'] * N)) * 1.0j
        for j in range(1, N + 1):
            for h in range(1, self.params['no_densities'] * N + 1):
                for i in range(1, self.params['no_densities'] + 1):
                    self.capital_o_matrix[i - 1, j - 1, h - 1] = \
                        sum([self.capital_b_2_matrix[N - 1, i - 1, m - 1] /
                             (self.v_vector[h - 1] + self.params['claim_size_lambda'][i - 1]) **
                             (m + 1.0 - l) *
                             (-z) ** (l - j) /
                             np.math.factorial(l - j) *
                             np.exp(self.params['claim_size_lambda'][i - 1] * z)
                             for m in range(j, N + 1)
                             for l in range(j, m + 1)])
        if self.params['debug']:
            print('capital_r_and_capital_o: Done!')

    def calculate(self, N, z, u):
        time_recorder = time.time()
        #################################################
        self.capital_a_and_capital_b(N)
        self.capital_m_and_capital_p(N, z)
        # self.capital_e_and_capital_e_asteric(N)
        self.roots_v(N)
        self.capital_r_and_capital_o(N, z)
        # self.roots_test(N)
        #################################################

        self.capital_y_vector = np.zeros(self.params['no_densities'] * N)
        for i in range(1, self.params['no_densities'] + 1):
            for j in range(1, N + 1):
                self.capital_y_vector[j - 1 + (i - 1) * N] = self.capital_p_matrix[i - 1, j - 1] - \
                                           z * self.capital_m_matrix[i - 1, j - 1]



        self.capital_x_matrix = np.zeros((self.params['no_densities']* N, self.params['no_densities'] * N)) * 1.0j
        for i in range(1, self.params['no_densities'] + 1):
            for j in range(1, N + 1):
                if j == N:
                    for k in range(1, self.params['no_densities'] * N + 1):
                        self.capital_x_matrix[j - 1 + (i - 1) * N, k - 1] = \
                                                        (self.capital_m_matrix[i - 1, j - 1] -
                                                         self.capital_o_matrix[i - 1, j - 1, k - 1]) * \
                                                        np.exp(self.v_vector[k - 1] * z)
                else:
                    for k in range(1, self.params['no_densities'] * N + 1):
                        self.capital_x_matrix[j - 1 + (i - 1) * N, k - 1] = \
                                                        (self.capital_m_matrix[i - 1, j - 1] +
                                                         self.capital_r_matrix[i - 1, j - 1, k - 1] -
                                                         self.capital_o_matrix[i - 1, j - 1, k - 1]) * \
                                                        np.exp(self.v_vector[k - 1] * z)
        self.capital_q_vector = np.linalg.solve(self.capital_x_matrix, self.capital_y_vector)
        self.result = sum(self.capital_q_vector * np.exp(self.v_vector * u))
        if self.params['debug']:
            print(time.time() - time_recorder)

        return self.result



        ## code trash

        # def modified_roots_v(self, N, coef_vec):
        #     self.real_root_mat = np.zeros((len(coef_vec), 2 * N))
        #     self.imag_root_mat = np.zeros((len(coef_vec), 2 * N))
        #     for j in range(len(coef_vec)):
        #         for k in range(N):
        #             coeff_list = [1.0,
        #                           self.params['claim_size_lambda'] - self.r,
        #                           self.params['claim_size_lambda'] *
        #                           (coef_vec[j] * self.params['poisson_process_lambda'] / self.params['premium_rate'] *
        #                            np.exp(-2j * k * math.pi / N) - self.r)]
        #             self.real_root_mat[j, 2 * k:(2 * k + 2)] = np.roots(coeff_list).real
        #             self.imag_root_mat[j, 2 * k:(2 * k + 2)] = np.roots(coeff_list).imag
        #     print('modifited_roots_v: Done!')

        # def m_summation(self, N, z, u):
        #     time_recorder = time.time()
        #     self.capital_a_and_capital_b(N)
        #     self.capital_m_and_capital_p(N, z)
        #     self.capital_m_summation = sum(self.capital_m_matrix *
        #                                    u ** (np.arange(1.0, N + 1.0) - 1.0) /
        #                                    np.array([np.math.factorial(item) for item in (np.arange(1.0, N + 1.0) - 1.0)]) *
        #                                    np.exp(-self.params['claim_size_lambda'] * u))
        #     print(time.time() - time_recorder)
        #     return self.capital_m_summation
        #
        # def E_asteric_summation_w(self, N, u):
        #     time_recorder = time.time()
        #     self.capital_a_and_capital_b(N)
        #     self.capital_e_and_capital_e_asteric(N)
        #     self.capital_e_asteric_summation_w = sum(self.capital_e_asteric_matrix_w *
        #                                              u ** (np.arange(1.0, N + 1.0) - 1.0) /
        #                                              np.array([np.math.factorial(item) for item in
        #                                                        (np.arange(1.0, N + 1.0)) - 1.0]) *
        #                                              np.exp(-u *
        #                                                     self.params['claim_size_lambda']))
        #     print(time.time() - time_recorder)
        #     return self.capital_e_asteric_summation_w
        #
        # def E_asteric_summation_1(self, N, u):
        #     time_recorder = time.time()
        #     self.capital_a_and_capital_b(N)
        #     self.capital_e_and_capital_e_asteric(N)
        #     self.capital_e_asteric_summation_1 = sum(self.capital_e_asteric_matrix_1 *
        #                                              u ** (np.arange(1.0, N + 1.0) - 1.0) /
        #                                              np.array([np.math.factorial(item) for item in
        #                                                        (np.arange(1.0, N + 1.0)) - 1.0]) *
        #                                              np.exp(-u *
        #                                                     self.params['claim_size_lambda']))
        #     print(time.time() - time_recorder)
        #     return self.capital_e_asteric_summation_1