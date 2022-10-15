import sys
import os
import multiprocessing as mp
import subprocess
from subprocess import Popen,PIPE
from datetime import date
import time


def ntuplizer(cmd_stagentuple_train, cmd_stagentuple_test, f_stdout, f_stderr):

    start2_time = time.time()
    subprocess.check_call(cmd_stagentuple_train, shell = True, stdout=f_stdout, stderr=f_stderr)
    subprocess.check_call(cmd_stagentuple_test, shell = True, stdout=f_stdout, stderr=f_stderr)
    end2_time = time.time()
    f_stdout.write("Stage_ntuple time (run only): {} \n".format(end2_time - end1_time))


if __name__ == '__main__':

    #paths
    inDIR='/eos/experiment/fcc/ee/generation/DelphesEvents/pre_fall2022_training/IDEA/'
    outDIR = "/eos/home-a/adelvecc/try_script_mp/"                                                                           ###NEEDS TO BE CHANGED###

    #create necessary subdirectories
    actualDIR = subprocess.check_output(["bash", "-c", "pwd"], universal_newlines = True)[:-1] #don't want '\n' character
    username = subprocess.check_output(["bash", "-c", "echo $USER"], universal_newlines=True)[:-1]  
    today = date.today().strftime("%Y%b%d")
    subdir = username + "_" + today
    subprocess.call(["bash", "-c", "mkdir -p {}".format(subdir)], cwd = outDIR)
    OUTDIR = outDIR + username + "_" + today + "/"
    print(OUTDIR)

    #set total number of events
    N = 1000000
    frac_split = 0.9 #training events fraction
    N_split = int(frac_split * N)

    #setup of the environment
    #subprocess.check_call(["source setup.sh"], shell = True, stdout = None, stderr = None)
    cmd_compile = "g++ -o MakeNtuple_constituents2 MakeNtuple_constituents2.cpp `root-config --cflags --libs` -Wall"
    subprocess.check_call(cmd_compile, shell = True, stdout=None, stderr=None)

    #create commands
    cmdbase_stage1 = 'fccanalysis run analysis_constituents_stage1_cluster.py '
    opt1_out = " --output {}stage1_ee_ZH_vvCLASS.root ".format(OUTDIR)
    opt1_in = " --files-list {}p8_ee_ZH_Znunu_HCLASS_ecm240/*.root ".format(inDIR)
    cmd_stage1 = cmdbase_stage1 + opt1_out + opt1_in
    
    cmdbase_stagentuple = "./MakeNtuple_constituents2 DIRstage1_ee_ZH_vvCLASS.root  DIRntuple_MOD_ee_ZH_vvCLASS.root "
    cmd_stagentuple = cmdbase_stagentuple.replace('DIR', OUTDIR)

    samples = [ 'bb', 'cc', 'ss', 'gg', 'qq']
    mods = ['train', 'test']

    #create files storing stdout and stderr
    list_stdout = [open(OUTDIR + "{}_stdout.txt".format(sample), "w") for sample in samples]
    list_stderr = [open(OUTDIR + "{}_stderr.txt".format(sample), "w") for sample in samples]

    ###=== RUN STAGE 1
    for i,sample in enumerate(samples):
        
        if(sample == 'qq'):
            cmd_stage1_f = cmdbase_stage1 + opt1_out.replace('CLASS', 'qq') + " --files-list {}p8_ee_ZH_Znunu_Huu_ecm240/*.root {}p8_ee_ZH_Znunu_Hdd_ecm240/*.root".format(inDIR, inDIR)
        else:
            cmd_stage1_f = cmd_stage1.replace('CLASS',sample)
        
        print(cmd_stage1_f)
        #run stage1
        start1_time = time.time()
        subprocess.check_call(cmd_stage1_f, shell = True, stdout=list_stdout[i], stderr=list_stderr[i])
        end1_time = time.time()
        list_stdout[i].write("Stage1 time: {} \n".format(end1_time - start1_time))

    ###=== RUN STAGE NTUPLE
    threads = []
    for i,sample in enumerate(samples):
        
        cmd_stagentuple_train = cmd_stagentuple.replace('CLASS',sample).replace('MOD', mods[0]) + " {} {} ".format(0, N_split)
        cmd_stagentuple_test = cmd_stagentuple.replace('CLASS',sample).replace('MOD', mods[1]) + " {} {} ".format(N_split, N)
        print(cmd_stagentuple_train)
        print(cmd_stagentuple_test)
            
        thread = mp.Process(target=ntuplizer, args=(cmd_stagentuple_train, cmd_stagentuple_test, list_stdout[i], list_stderr[i]))
        thread.start()
        threads.append(thread)
        
    for proc in threads:
        proc.join()

    for i in range(len(list_stdout)):
        f_stdout[i].close()
        f_stderr[i].close()
   
    
#1) How to improve?
