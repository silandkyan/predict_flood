# predict_flood

## Manage environment
1. Initial environment creation:

`conda create --name geo`
`conda activate geo`

2. Go to project home folder and generate environment file (needs to be updated after installation of additional packages!): 

`conda env export > geo.yml`

3. to create the environment from the file, run:

`conda env create -f geo.yml`

4. and activate:

`conda activate geo`

5. to verify:

`conda env list`

6. Make new ipykernel for running the notebook on:

`python -m ipykernel install --user --name geo --display-name "Python (geo)"`

7. Finally, activate this kernel in the notebook settings.
