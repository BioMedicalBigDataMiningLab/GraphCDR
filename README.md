# GraphCDR
Source code for "GraphCDR: Contrastive graph neural network for cancer drug response prediction"

# Requirements
* Python == 3.6
* PyTorch == 1.4
* PyTorch Geometry == 1.6
* hickle >= 2.1.0

# Usage
python graphCDR.py \<parameters\>
  
# Predicted missing data
As GDSC database only measured IC50 of part cell line and drug paires. We applied GraphCDR to predicted the missing IC50 values. The predicted results can be find at data/Missing_pairs.txt. 
