import numpy as np
import sympy
import os
import sys
import csv


def read(filename):
    with open(os.path.join(sys.path[0], filename), "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        lines = np.array([i for i in reader])

    return header, lines


def solve(lipid,a,b,c,d):
    '''
    lipid (string): helper lipid
    a (float): Dlin:lipid
    b (float): Chol:DMG
    c (float): N:P ratio
    d (float): lipid+Dlin+(d% pDNA) ~default = 80
    100-d (float): Chol+DMG+(e% pDNA) ~default = 20

    '''
    n = 0 #presence of ionizable N
    if lipid == 'DOTAP':
        lipid_wt = 663.1
        n = 1
    elif lipid == 'DOPE':
        lipid_wt = 744.03
    elif lipid == 'DSPC':
        lipid_wt = 790.15
    elif lipid == '18PG':
        lipid_wt = 799.042

    N = np.array([[-a,1,0,0,0,0],
                [0,0,1,-b,0,0],
                [n,1,0,0,-c*9304,0],
                [1,1,0,0,d/100,d],
                [0,0,1,1,(100-d)/100,(100-d)]])

    mol_solutions = np.array(sympy.Matrix(N).rref()[0])[:,5]

    def conversion(mol_percents, lipid_wt):
        
        Dlin = 642.11
        Chol = 386.65
        DMG = 2509
        pDNA = 3023800

        MW = np.array([lipid_wt, Dlin, Chol, DMG, pDNA])
        wts = mol_percents*MW
        wt_percents = wts/sum(wts)*100

        return wt_percents

    wt_solutions = conversion(mol_solutions, lipid_wt)

    return mol_solutions, wt_solutions


def write(solved_data):
    
    with open(os.path.join(sys.path[0], "masterformulas_calculated.csv"), "w", newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerows(solved_data)

    return


if __name__ == '__main__':
    
    _, data = read("masterformulas.csv")
    #print(data)
    
    solved_data = np.array(['Formula label', 'Helper lipid', 'Lipid wt%', 'Dlin wt%', 'Chol wt%', 'DMG wt%', 'pDNA wt%', 'RLU in HepG2', 'RLU in N2a', 'RLU in ARPE19'])
    for row in data:
        _, wt = solve(row[1], float(row[3]), float(row[5]), float(row[2]), float(row[4]))
        new_row = np.concatenate((row[:2], wt, row[6:]))
        solved_data = np.vstack([solved_data, new_row])

    #print(solved_data)
    write(solved_data)