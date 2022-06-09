import torch
import numpy
import sys, os, time
import yaml, atexit
from ._version import __version__, __git_sha__

from simple_nn.init_inputs import initialize_inputs
from simple_nn.models import train


def run(input_file_name):
    print('Meta-SIMPLE-NN')
    start_time = time.time()

    logfile = None
    logfile = open('LOG', 'w', 1)
    atexit.register(_close_log, logfile)
    inputs = initialize_inputs(input_file_name, logfile)

    seed = inputs['random_seed']
    torch.manual_seed(seed)
    numpy.random.seed(seed)
    _log_header(inputs, logfile)

    train(inputs, logfile)

    logfile.write(f"Total wall time: {time.time()-start_time} s.\n")

def _close_log(logfile):
    logfile.flush()
    os.fsync(logfile.fileno())
    logfile.close()

def _log_header(inputs, logfile):
    # TODO: make the log header (low priority)
    logfile.write("SIMPLE_NN v{0:} ({1:})".format(__version__, __git_sha__))
    logfile.write("{:>50}: {:>10}\n".format("SEED", inputs["random_seed"]))
    logfile.write("{}\n".format('-'*88))
    logfile.write("{:^88}\n".format("  _____ _ _      _ _ ___  _     _____       __    _ __    _ "))
    logfile.write("{:^88}\n".format(" / ____| | \    / | '__ \| |   |  ___|     |  \  | |  \  | |"))
    logfile.write("{:^88}\n".format("| |___ | |  \  /  | |__) | |   | |___  ___ |   \ | |   \ | |"))
    logfile.write("{:^88}\n".format(" \___ \| |   \/   |  ___/| |   |  ___||___|| |\ \| | |\ \| |"))
    logfile.write("{:^88}\n".format(" ____| | | |\  /| | |    | |___| |___      | | \   | | \   |"))
    logfile.write("{:^88}\n".format("|_____/|_|_| \/ |_|_|    |_____|_____|     |_|  \__|_|  \__|"))
    logfile.write("{:^88}\n".format("                ___         ___"))
    logfile.write("{:^88}\n".format('                /   \       /   \\'))
    logfile.write("{:^88}\n".format("                \ ( /_______\ ) /"))
    logfile.write("{:^88}\n".format('                /               \\'))
    logfile.write("{:^88}\n".format('                /  _          _   \\'))
    logfile.write("{:^88}\n".format("    /^/         (  /            \   )"))
    logfile.write("{:^88}\n".format("   / /___       )                   ("))
    logfile.write("{:^88}\n".format('   (  ____)     /     0  ____  0      \\'))
    logfile.write("{:^88}\n".format("   (  ____)    (    __  / <> \  __     )"))
    logfile.write("{:^88}\n".format("   (  ____)     \    \__________/     /"))
    logfile.write("{:^88}\n".format("  (______)      \    \________/     /"))
    logfile.write("{:^88}\n".format("               \_________________/"))
    logfile.write("\n")
