# GraphCDR
Source code and data for "GraphCDR: Contrastive graph neural network for cancer drug response prediction"

![Framework of GraphCDR](https://github.com/liuxuan666/GraphCDR/blob/main/Framework.png)  

# Requirements
* Python >= 3.6
* PyTorch >= 1.4
* PyTorch Geometry >= 1.6
* hickle >= 3.4
* DeepChem >= 2.4
* RDkit >= 2020.09

# Usage
python graphCDR.py \<parameters\>
python graphCDR-ccle.py \<parameters\>
  
# Case study (predicted missing response pairs)
As GDSC database only measured IC50 of part cell line and drug pairs. We applied GraphCDR to predicted the missing types of responses. The predicted results can be find at data/Case study (missing pairs).xlsx
