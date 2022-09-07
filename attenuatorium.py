import math
import os
import tkinter as tk
import numpy as np
import scipy
import scipy.stats
from tkinter import filedialog

class reflection():
    def __init__(self) -> None:
        self.ind = (0,0,0)
        self.intensity = 0.0
        self.sigma = 0.0
    def __init__(self, h, k, l, i, s) -> None:
        self.ind = (h,k,l)
        self.intensity = i
        self.sigma = s
    def l(self) -> int:
        return self.ind[2]
    def k(self) -> int:
        return self.ind[1]
    def h(self) -> int:
        return self.ind[0]
    def i(self) -> float:
        return self.intensity
    def s(self) -> float:
        return self.sigma
    def index(self) -> tuple:
        return self.ind
    def i_over_s(self) -> float:
        return self.intensity / self.sigma
    def ratio(self, compare) -> float:
        return self.i() / compare.i()
    def sigma_ratio(self, compare) -> float:
        return self.s() / compare.s()

class cell():
    def __init__(self) -> None:
        self.a = self.b = self.c = 5
        self.alpha = self.beta = self.gamma = 90
        self.ca = self.cb = self.cg = 1.0
        self.sa = self.sb = self.sg = 0.0
        self.wavelength = 0.71
    def __init__(self,a,b,c,alpha,beta,gamma,wl) -> None:
        self.a = float(a)
        self.b = float(b)
        self.c = float(c)
        self.alpha = float(alpha)
        self.beta  = float(beta)
        self.gamma = float(gamma)
        self.wavelength = float(wl)
        self.ca = math.cos(math.pi/180.0*self.alpha)
        self.cb = math.cos(math.pi/180.0*self.beta)
        self.cg = math.cos(math.pi/180.0*self.gamma)
        self.sa = math.sin(math.pi/180.0*self.alpha)
        self.sb = math.sin(math.pi/180.0*self.beta)
        self.sg = math.sin(math.pi/180.0*self.gamma)
    def get_d_of_hkl(self,hkl) -> float:
        upper = pow(hkl[0], 2) * pow(self.sa, 2) / pow(self.a, 2) \
              + pow(hkl[1], 2) * pow(self.sb, 2) / pow(self.b, 2) \
              + pow(hkl[2], 2) * pow(self.sg, 2) / pow(self.c, 2) \
              + 2.0 * hkl[1] * hkl[2] / (self.b * self.c) * (self.cb * self.cg - self.ca) \
              + 2.0 * hkl[0] * hkl[2] / (self.a * self.c) * (self.cg * self.ca - self.cb) \
              + 2.0 * hkl[0] * hkl[1] / (self.a * self.b) * (self.ca * self.cb - self.cg)
        lower = 1 - pow(self.ca, 2) - pow(self.cb, 2) - pow(self.cg, 2) + 2 * self.ca * self.cb * self.cg
        d = math.sqrt(lower / upper)
        return d
    def get_stl_of_hkl(self,hkl) -> float:
        return 1.0 / (2 * self.get_d_of_hkl(hkl))

class reflection_list():
    def __init__(self) -> None:
        self.refl_list = []
        self.min_d = 0.0
        self.max_d = 0.0
    def append_refl(self,ref) -> None:
        if type(ref) != type(reflection):
            print("Mismatch in types!")
            exit(-1)
        self.refl_list.append(ref)
    def append(self, h, k, l, i, s) -> None:
        self.refl_list.append(reflection(h,k,l,i,s))
    def index_tuple(self) -> tuple:
        ret = []
        for ref in self.refl_list:
            ret.append(tuple(ref.index()))
        return tuple(ret)
    def index_set(self) -> set:
        li = self.index_tuple()
        ret = set(li)
        return ret
    def get_hkl(self,h,k,l) -> list:
        ret = []
        for ref in self.refl_list:
            if ref.h() == h and ref.k() == k and ref.l() == l:
                ret.append(ref)
        return ret
    def has_hkl(self,h,k,l) -> bool:
        ret = False
        for ref in self.refl_list:
            if ref.h() == h and ref.k() == k and ref.l() == l:
                ret = True
        return ret
    def get_subset(self, hkl_list):
        ret = reflection_list()
        for ref in self.refl_list:
            if ref.ind in hkl_list:
                ret.append_refl(ref)
        return ret
    def get_with_i_to_s_bigger_than(self, cutoff):
        ret = reflection_list()
        for ref in self.refl_list:
            if ref.i_over_s() > cutoff:
                ret.append(ref)
        return ret
    def count_i_to_s_bigger_than(self, cutoff) -> int:
        ret = 0
        for ref in self.refl_list:
            if ref.i_over_s() > cutoff:
                ret+=1
        return ret
    def count_i_to_s_bigger_than(self, cutoff) -> int:
        if type(cutoff) == type(int):
            ret = 0
            for ref in self.refl_list:
                if ref.i_over_s() > cutoff:
                    ret+=1
        if type(cutoff) == type([]) or type(cutoff) == type(()):
            ret = []
            for cut in cutoff:
                ret.append(0)
                for ref in self.refl_list:
                    if ref.i_over_s() > cut:
                        ret[-1]+=1
        return ret
    def size(self) -> int:
        return len(self.refl_list)
    def set_cell(self,a,b,c,alpha,beta,gamma, wl) -> None:
        self.cell = cell(a,b,c,alpha,beta,gamma,wl)
    def get_min_d(self):
        return self.min_d
    def get_max_d(self):
        return self.max_d



def read_hkl(filename) -> reflection_list:
    file = open(filename).readlines()
    listy = reflection_list()
    for i,line in enumerate(file):
        if "   0   0   0    0.00    0.00   0" in line:
            last_line = i
            break
        listy.append(
              int(line[1:4]),
              int(line[5:8]),
              int(line[9:12]),
              float(line[13:20]),
              float(line[21:28])
            )
    for n in range(len(file)-i):
        if "CELL" in file[n+i]:
            dump,wl,a,b,c,alpha,beta,gamma = file[n+i].split()
            listy.set_cell(a,b,c,alpha,beta,gamma,wl)
    return listy

root = tk.Tk()
root.withdraw()

file_paths = filedialog.askopenfilenames()

number_of_file = len(file_paths)

sets = []
hkls = []
intersects = []

for i,file in enumerate(file_paths):
    sets.append(read_hkl(file))
    file2 = os.path.basename(file)
    nr = sets[i].size()
    ios_counts = sets[i].count_i_to_s_bigger_than((4.0,3.0,2.0,1.0))
    print(f"{file2:15s} has {nr} reflections, {ios_counts[0]/nr*100:4.1f}%/{ios_counts[1]/nr*100:4.1f}%/{ios_counts[2]/nr*100:4.1f}%/{ios_counts[3]/nr*100:4.1f}% with I/s bigger than 4/3/2/1")
    hkls.append(sets[-1].index_set())
    if i > 0:
        intersects.append(hkls[i-1].intersection(hkls[i]))


print("OK")