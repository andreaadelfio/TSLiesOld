import glob
import os
import tqdm
import multiprocessing

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

def compare_folders(folder_1, folder_2):
    files_1 = set(os.path.basename(file) for file in glob.glob(folder_1 + '/*'))
    files_2 = set(os.path.basename(file) for file in glob.glob(folder_2 + '/*'))
    print('Files in folder 1:', len(files_1))
    print('Files in folder 2:', len(files_2))
    print('Files only in folder 1:', len(files_1 - files_2))
    print('Files only in folder 2:', len(files_2 - files_1))
    with open('missing_files.txt', 'w') as f:
        for file in sorted(files_2 - files_1):
            f.write(file + '\n')

def copy_files(src, dst):
    os.makedirs(dst, exist_ok=True)
    with open('out.txt', 'r') as f:
        folders = []
        for folder in f.readlines():
            folders.append(folder[:-1])
    for folder in tqdm.tqdm(folders, desc='Copying folders'):
        # copies directory structure
        os.makedirs(os.path.join(dst, folder), exist_ok=True)
        for file in glob.glob(os.path.join(src, folder) + '/*'):
            # print(file, os.path.join(dst, folder))
            os.system(f'cp "{file}" "{os.path.join(dst, folder)}"')

# dirname = '/mnt/E888C7FF88C7C9F0/Users/andad/Downloads/newRuns_v16'
# newdirname = './data/LAT_ACD/raw/newRuns_v16'
folder_1 = 'data/LAT_ACD/processed/new/pk'
folder_2 = 'data/LAT_ACD/processed/new_old/pk'
# # extractor(dirname, newdirname)
# # confront(dirname)
compare_folders(folder_1, folder_2)
def process_file(file):
    command = 'scp slacd:/sdf/home/m/maldera/fermi-user/newRuns_v2/'
    file = file.split('\n')[0]
    os.makedirs('/home/andrea-adelfio/OneDrive/Workspace INFN/ACDAnomalies/data/LAT_ACD/raw/maldera/' + file + '/', exist_ok=True)
    os.system(command + file + '/*.root "/home/andrea-adelfio/OneDrive/Workspace INFN/ACDAnomalies/data/LAT_ACD/raw/maldera/' + file + '/"')

def copy_files_from_folder(file):
    os.makedirs('/home/andrea-adelfio/OneDrive/Workspace INFN/ACDAnomalies/data/LAT_ACD/raw/for_alberto/' + file + '/', exist_ok=True)
    for files in glob.glob('/home/andrea-adelfio/OneDrive/Workspace INFN/ACDAnomalies/data/LAT_ACD/raw/new_with_correct_triggs/' + file + '/*'):
        os.system('cp "' + files + '" "/home/andrea-adelfio/OneDrive/Workspace INFN/ACDAnomalies/data/LAT_ACD/raw/for_alberto/' + file + '/"')



def confront_lists_in_two_files(file_1, file_2):
    with open(file_1, 'r') as f:
        files_1 = set(f.readlines())
    with open(file_2, 'r') as f:
        files_2 = set(f.readlines())
    print('Files in file 1:', len(files_1))
    print('Files in file 2:', len(files_2))
    print('Files only in file 1:', (files_1 - files_2))

    # with multiprocessing.Pool() as pool:
    #     pool.map(process_file, files_1 - files_2)

# src = '/home/andrea-adelfio/OneDrive/Workspace INFN/ACDAnomalies/data/LAT_ACD/raw/new/'
# dst = '/home/andrea-adelfio/OneDrive/Workspace INFN/ACDAnomalies/data/LAT_ACD/raw/new_with_correct_triggs/'
# copy_files(src, dst)

# file_1 = 'utils/maldera.txt'
# file_2 = 'utils/miei.txt'
# confront_lists_in_two_files(file_1, file_2)


# file = 'out.txt'
# with open(file, 'r') as f:
#     files = []
#     for file in f.readlines():
#         files.append(file.split('\n')[0])
#     with multiprocessing.Pool() as pool:
#         pool.map(copy_files_from_folder, files)
