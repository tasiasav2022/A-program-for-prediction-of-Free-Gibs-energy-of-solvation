'''
A neural network to predict solvation free energy in organic solvents
Authors: Anastasia Savchenko and Sergei Vyboishchikov
Universitat de Girona, July 2026
The program is used from the command line:
SolvPredict.py solute_smiles1 solute_smiles2 ... solvent
Required modules: RDKit, Numpy, and Pickle
'''
import sys, pickle, numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

if len(sys.argv) <= 1:
   print("Usage: SolvPredict.py solute_smiles1 solute_smiles2 ... solvent")
   quit()

solute_smiles = sys.argv[1:-1]
solvent = sys.argv[-1]

pickle_filename = "ModelData.pkl"
try:
   SolventDict,model,nCounts,radius = pickle.load(open(pickle_filename,"rb"))
except:
   print('Model file "'+pickle_filename+'" does not exist, cannot be opened, or has an incorrect format')
   quit()

if solvent not in SolventDict:
   print('Solvent "'+solvent+'" not supported')
   quit()

MorganMatrix = np.zeros((len(solute_smiles),nCounts),dtype='int')
MistakeVector = np.zeros(MorganMatrix.shape[0],dtype='int')
for i, smi in enumerate(solute_smiles):
   mol = Chem.MolFromSmiles(smi)
   if mol is None:
      print('Warning: Could not parse SMILES "'+smi+'". Row will remain zeros.')
      MistakeVector[i] = 1
   else:
      molH = Chem.AddHs(mol)
      fp = AllChem.GetHashedMorganFingerprint(molH, radius=radius, nBits=nCounts).GetNonzeroElements()
      for entorn_id,count in fp.items():
         MorganMatrix[i,entorn_id] = count
        
SolventFeaturesMatrix = np.tile(SolventDict[solvent], (len(solute_smiles),1))
Predictions = model.predict(np.hstack((MorganMatrix,SolventFeaturesMatrix)))

print("\n"+50*"="+"\nRESULTS AND SOLVENT PARAMETERS\n"+50*"=")
for i, (smi,delta_g_solv) in enumerate(zip(solute_smiles, Predictions)):
   print("Solute SMILES: "+smi+"\nSolvent: "+solvent)
   if MistakeVector[i]:
      print("Predicted Value: None (mistake in SMILES)")
   else:
      print("Predicted Value: %6.2f kcal/mol"%delta_g_solv)
   print(50*"-")
