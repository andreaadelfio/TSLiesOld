import glob
import os
import tqdm

def extractor(dirname, newdirname):
    files = glob.glob(dirname + '/*/*/*.root')
    folders = glob.glob(newdirname + '/*')
    os.makedirs(newdirname, exist_ok=True)
    for file in tqdm.tqdm(files, desc='Extracting roots'):
        os.makedirs(os.path.join(newdirname, os.path.basename(file).split('_')[1]), exist_ok=True)
        os.system('cp ' + file + ' ' + os.path.join(newdirname, os.path.basename(file).split('_')[1]))
        # os.system('mv ' + file + ' ' + os.path.join(newdirname, os.path.basename(file).split('_')[1]))

def confront(dirname):
    folders = glob.glob(dirname + '/*/*/*.root')
    files = set()
    with open(os.path.join(dirname, 'ANDREA_up1.csv'), 'r') as f:
        for file in f.readlines():
            if 'QUI' in file:
                break
            files.add(file[:-1])
    folders_set = set([os.path.basename(folder).split('_')[1] for folder in folders])
    print('Created runs', len(folders_set), '; missing runs:', len(files))
    with open('failed_runs.txt', 'w') as f:
        for file in sorted(files - folders_set):
            f.write(file + '\n')

dirname = '/mnt/E888C7FF88C7C9F0/Users/andad/Downloads/newRuns_v13'
newdirname = './data/LAT_ACD/raw/newRuns_v7_extracted_v3'
extractor(dirname, newdirname)
# confront(dirname)
