import math
import os
import tkinter as tk
import numpy as np
import scipy
import scipy.stats
from tkinter import filedialog
import matplotlib.pyplot as plt

class reflection():
    def __init__(self) -> None:
        self.ind = (0,0,0)
        self.intensity = 0.0
        self.sigma = 0.0
    def __init__(self, hkl, i, s) -> None:
        self.ind = hkl
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
        d = np.sqrt(lower / upper)
        return d
    def get_stl_of_hkl(self,hkl) -> float:
        return 1.0 / (2 * self.get_d_of_hkl(hkl))

class reflection_list():
    def __init__(self) -> None:
        self.refl_list = np.array([[],[],[]])
        self.min_d = 0.0
        self.max_d = 0.0
        self.name = ""
    def append(self, h, k, l, i, s) -> None:
        t = np.array([[(h,k,l),i,s]],dtype=object)
        self.refl_list = np.append(self.refl_list,t.transpose(),axis=1)
    def index_tuple(self) -> tuple:
        return self.refl_list[0]
    def unique_hkl(self) -> set:
        li = self.index_tuple()
        ret = np.unique(li)
        return ret
    def get_hkl(self,hkl) -> list:
        ret = []
        #condition = np.array(self.refl_list[0]==hkl)
        condition = np.array([x == hkl for x in self.refl_list[0]])
        ret = np.compress(condition,self.refl_list,axis=1)
        return ret
    def has_hkl(self,hkl) -> bool:
        ret = False
        for ref in self.refl_list:
            if ref.index() == hkl:
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
        ret = np.count_nonzero(self.refl_list[1]/self.refl_list[2] > cutoff)
        return ret
    def count_i_to_s_bigger_than(self, cutoff) -> int:
        if type(cutoff) == type(int):
            ret = np.count_nonzero(self.refl_list[1]/self.refl_list[2] > cut)
        if type(cutoff) == type([]) or type(cutoff) == type(()):
            ret = []
            for cut in cutoff:
                ret.append(np.count_nonzero(abs(self.refl_list[1]/self.refl_list[2]) > cut))
        return ret
    def size(self) -> int:
        return self.refl_list[0].size
    def set_cell(self,a,b,c,alpha,beta,gamma, wl) -> None:
        self.cell = cell(a,b,c,alpha,beta,gamma,wl)
        mind = 999.0
        maxd = 0.0
        d_func = np.vectorize(self.cell.get_d_of_hkl)
        d = d_func(self.refl_list[0])
        mind = np.min(d)
        maxd = np.max(d)
        self.max_d = maxd
        self.min_d = mind
    def get_min_d(self):
        return self.min_d
    def get_max_d(self):
        return self.max_d
    def get_intersection_indices(self, given):
        return np.intersect1d(self.refl_list[0],given.refl_list[0])



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
    if last_line < len(file) - 1:
        for n in range(len(file)-last_line):
            if "CELL" in file[n+last_line]:
                dump,wl,a,b,c,alpha,beta,gamma = file[n+last_line].split()
                listy.set_cell(a,b,c,alpha,beta,gamma,wl)
    listy.name = os.path.basename(filename)
    return listy

root = tk.Tk()
root.withdraw()

file_paths = filedialog.askopenfilenames()

number_of_file = len(file_paths)

sets = []
hkls = []
intersects = []
colors = ['b','g','r','y']
names = []

for i,file in enumerate(file_paths):
    sets.append(read_hkl(file))
    names.append(sets[-1].name)
    unique_indices = sets[-1].unique_hkl()
    nr_unique_indices = unique_indices.size
    nr = sets[-1].size()
    ios_counts = sets[-1].count_i_to_s_bigger_than((4.0,3.0,2.0,1.0))
    print(f"{sets[-1].name:15s} has {nr} reflections, {ios_counts[0]/nr*100:4.1f}%/{ios_counts[1]/nr*100:4.1f}%/{ios_counts[2]/nr*100:4.1f}%/{ios_counts[3]/nr*100:4.1f}% \
with I/s bigger than 4/3/2/1 d_min: {sets[-1].get_min_d():4.2f} d_max: {sets[-1].get_max_d():4.2f}; {nr_unique_indices:4d} unique indices with avg. \
{nr/nr_unique_indices:5.3f} redundancy")
    hkls.append(unique_indices)
    if i > 0:
        intersects.append(sets[-2].get_intersection_indices(sets[-1]))
        _x = []
        _y = []
        for hkl in intersects[-1]:
            refs1 = sets[-2].get_hkl(hkl)
            refs2 = sets[-1].get_hkl(hkl)
            av1 = 0
            size1 = refs1.shape[1]
            for r in range(size1):
                av1 += refs1[1][r]
            av1 /= size1
            _x.append(av1)
            av2 = 0
            size2 = refs2.shape[1]
            for r in range(size2):
                av2 += refs2[1][r]
            av2 /= len(refs2)
            _y.append(av2)
        plt.scatter(_x,_y,s=5,facecolors='none',edgecolors=colors[i],label=f"{sets[-2].name} vs {sets[-1].name}")
plt.xlabel(f"I(more attenuation)")
plt.ylabel(f"I(less attenuation)")
plt.legend(loc="upper left")
plt.show()

fig,axes = plt.subplots(2,1)
axes[0].invert_xaxis()
a = []
axes[0].set_xlabel("d /Angs")
axes[1].set_xlabel("I/sigma")
axes[0].set_ylabel("I")
axes[1].set_ylabel("# refl.")
for i,s in enumerate(sets):
    x = []
    y = []
    i_o_s = []
    for ref in s.refl_list:
        x.append(s.cell.get_d_of_hkl(ref.index()))
        y.append(ref.i())
        i_o_s.append(ref.i_over_s())
    axes[0].scatter(x,y,s=5,facecolors='none',edgecolors=colors[i],label=names[i])
    a.append(np.array(i_o_s))
    
    #axes.scatter(x,b_result,s=10,facecolors='none',edgecolors='g',label="b")
    #axes.scatter(x,c_result,s=10,facecolors='none',edgecolors='r',label="c")
    #axes.scatter(x,d_result,s=10,facecolors='none',edgecolors='y',label="d")
    #axes.plot(x,g_result,'--',label="f_table1")
upper = int(max(np.amax(x) for x in a))
steps = int(upper/2)
axes[1].hist(a,bins=np.linspace(0,upper,steps), color=colors, label=names)
plt.legend(loc="upper right")
plt.show()


print("OK")