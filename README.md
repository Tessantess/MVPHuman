# MVPHuman Dataset 

![](teaser/teaser.png)

## Installation
1. Install the environment and the SMPL model follow [SCANimate](https://arxiv.org/pdf/2104.03313).
2. Install [pypoisson](https://github.com/mmolero/pypoisson). 
## Data Processing   

Note that SCANimate requires accurate fitted SMPL to generate natural canonical mesh. Thus, we recommend you selects the scans with simple poses such as Action01 (A-pose) or Action03 (T-pose)
to get the canonical mesh.   

**0. Download extra data from this [link](https://drive.google.com/file/d/1zoCojdsrrAHRPif2J2b79jNoKphKYgWw/view?usp=sharing).**

​	Put data in the folder `./data/smpl_related/smpl_data`. 
    
**1. Fit SMPL for raw scan.**  
```sh 
$ python -m apps.fit_smpl.py --in_dir ./data_example --out_dir ./data_example  
```
​		This script will create generate results in folder `out_dir/subject_id/smpl`.
 
**2. Run SCANimate.** 
 ```sh
$ python -m apps.run_scanimate --in_dir ./data_example --out_dir ./data_example  
```
​		This script will create generate results in folder `out_dir/subject_id/cano`.
 
  
## Render Dancing Video (coming soon) 

## Download Instructions

## Acknowledgments
 
 Thanks to the authors of SCANimate. 


 
 
  