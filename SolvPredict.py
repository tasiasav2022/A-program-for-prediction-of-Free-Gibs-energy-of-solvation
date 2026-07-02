import sys, os, pickle, numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

if len(sys.argv) <= 1:
   print("Usage: SolvPredict.py smiles1 smiles2 ... solvent")
   quit()

input_smiles = sys.argv[1:-1]
solvent = sys.argv[-1]

pickle_filename = "ModelData.pkl"
if os.path.exists(pickle_filename):
   with open(pickle_filename, "rb") as f:
      SolventDict,model,nCounts,radius = pickle.load(f)
else:
   print("Model file "+pickle_filename+" does not exist")  
   quit()

if solvent not in SolventDict:
   print('Solvent "'+solvent+'" not supported')
   quit()

MorganMatrix = np.zeros((len(input_smiles),nCounts)) 
for i, smi in enumerate(input_smiles):
   mol = Chem.MolFromSmiles(smi)
   if mol is not None:
      molH = Chem.AddHs(mol)
      fp = AllChem.GetHashedMorganFingerprint(molH, radius=radius, nBits=nCounts).GetNonzeroElements()
      for entorn_id,count in fp.items():
         MorganMatrix[i,entorn_id] = count
   else:
      print('Warning: Could not parse SMILES "'+smi+'". Row will remain zeros.')
        
SolventFeatures = SolventDict[solvent]
SolventFeaturesMatrix = np.tile(SolventFeatures, (len(input_smiles),1))

X = np.hstack((MorganMatrix,SolventFeaturesMatrix))
Predictions = model.predict(X)

print("\n"+50*"="+"\nRESULTS AND SOLVENT PARAMETERS\n"+50*"=")

for smi, delta_g_solv, fp in zip(input_smiles, Predictions, MorganMatrix):
   print("Solute SMILES: "+smi)
   print("Solvent: "+solvent)
   if np.max(fp)<0.5:
      print("Predicted Value: None (error in SMILES)")
   else:
      print(f"Predicted Value: {delta_g_solv:6.2f} kcal/mol")
   print(50*"-")
