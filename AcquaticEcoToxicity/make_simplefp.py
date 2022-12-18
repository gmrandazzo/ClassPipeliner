#!/usr/bin/env python3
"""
make_simplefp.py 

Copyright (C) <2022>  Giuseppe Marco Randazzo <gmrandazzo@gmail.com>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

This is a very simple fingerprint based on atom/bonds

Algorhtm:

1) Read the csv smi, activity file

2) Iterate trough atoms and apply the hash function to get the unique integer
   that map the atom and the bond.

"""
import sys
from openbabel import openbabel

def smi2obmol(smiles):
    """
    Read a smiles string and conver into mol objec
    """
    conv = openbabel.OBConversion()
    conv.SetInAndOutFormats("smi", "mol")
    mol = openbabel.OBMol()
    conv.ReadString(mol, smiles)
    return mol


def simple_fp(mol):
    """
    Calculate a fingerprint that maps each type of atom/bond to a unique integer
    """
    fp = [0] * 1024
    # iterate over atoms
    for atom in openbabel.OBMolAtomIter(mol):
        atom_code = hash(atom.GetType()) % 1024
        fp[atom_code] = 1
    # iterate over bonds
    for bond in openbabel.OBMolBondIter(mol):
        bond_code = hash(bond.GetBondOrder()) % 1024
        fp[bond_code] = 1
    return fp


def simple_fp_with_neighbour(mol):
    """
    Calculate a fingerprint that maps each type of atom/bond to a unique integer
    with neighbours atom contribution.
    """
    fp = [0] * 1024    
    for atom in openbabel.OBMolAtomIter(mol):
        atom_code = hash(atom.GetType()) % 1024
        fp[atom_code] = 1
        for neighbour_atom in openbabel.OBAtomAtomIter(atom):
            atom_code = hash(atom.GetType()+neighbour_atom.GetType()) % 1024
            fp[atom_code] = 1
            atom_code = hash(neighbour_atom.GetType()+atom.GetType()) % 1024
            fp[atom_code] = 1
    return fp


def main():
    if len(sys.argv) != 4:
        print("Usage %s [name,smi,act file] [fp type] [desc, act file]" % (sys.argv[0]))
    else:
        fi = open(sys.argv[1], "r")
        fo = open(sys.argv[3], "w")
        for line in fi:
            if "Molecule" in line:
                continue
            else:
                fp_fun = None
                if sys.argv[2] == "simple":
                    fp_fun = simple_fp
                else:
                    fp_fun = simple_fp_with_neighbour
                v=str.split(line.strip(), ",")
                if len(v) == 3:
                    mol = smi2obmol(v[1])
                    fp = fp_fun(mol)
                    activity=None
                    if float(v[2]) < 0.001:
                        activity = 1
                    else:
                        activity = 0
                    fo.write("%s" % (v[0]))
                    for i in range(len(fp)):
                        fo.write(",%d" % (fp[i]))
                    fo.write(",%d\n" % (activity))
                else:
                    continue
        fo.close()
        fi.close()
    return 0

if __name__ in "__main__":
    main()
