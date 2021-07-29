'''
most of the code from github.com/lynetcha
'''
import os
import sys
import time
import re
import subprocess

import numpy as np
import torch
import torch.nn as nn
import torch.onnx
from PIL import Image
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torchvision.models import resnet

import argparse
import utils
from Network import *

base_models = {'resnet50': (resnet.resnet50, 2048),
               'resnet34': (resnet.resnet34, 512),
               'resnet18': (resnet.resnet18, 512)}
ALLTYPES = ['b', 'c', 'f', 'f2', 'f3']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_TYPE_COUNT = {'b':3, 'c':4, 'f':7, 'f2':3, 'f3':5}

MYMAT="mymat"
result_folder = "/cvgl2/u/monaavr/nvidia_mdl_templates/nvidia/vMaterials"
samplefolder = "model_output/predictions"
samplepath = "model_output/predictions/mymat.txt"
pythonpath = "/cvgl2/u/lynetcha/programs/python/py3.6.10/lib"
DATA_PATH = "/cvgl/group/vMaterials2"

allmats = open("data/vMaterials/matlist.txt").read().splitlines()


'''
code to cache predictions
'''

def cache_pred(pred):
    if not os.path.isdir(samplefolder):
        os.makedirs(samplefolder)
    with open(samplepath, "w") as f:

        for key in pred:
            pred[key] = pred[key].squeeze().tolist()
        f.write("%d\n" % pred["classid"])

        for ix in range(MAX_TYPE_COUNT['b']):
            f.write("%d\n" % pred["b"][ix])
        for ix in range(MAX_TYPE_COUNT['c']):
            f.write("%f %f %f\n" % (pred['c'][ix][0],pred['c'][ix][1], pred['c'][ix][2]))
        for ix in range(MAX_TYPE_COUNT['f']):
            f.write("%f\n" % pred["f"][ix])
        for ix in range(MAX_TYPE_COUNT['f2']):
            f.write("%f %f\n" % (pred["f2"][ix][0],pred["f2"][ix][1]))
        for ix in range(MAX_TYPE_COUNT['f3']):
            f.write("%f %f %f\n" % (pred["f3"][ix][0],pred["f3"][ix][1], pred["f3"][ix][2]))


'''
code to recreate the mdl from cached text predictions
'''

def load_norm_ranges(fname):
    normdict = {}
    lines = [k.split(',') for k in open(fname).read().splitlines()]
    for line in lines:
        name = line[0]
        normdict[name] = {}
        for param in line[1:]:
            pname, xmin, xmax = param.split(" ")
            normdict[name][pname] = [xmin, xmax]
    return normdict


def replace_mat_name(output, oldmat, newmat):
    count = 0
    lines = output.splitlines()
    for i in range(len(lines)):
        if "export" in lines[i] and oldmat in lines[i]:
            lines[i] = lines[i].replace(oldmat, newmat)
            count += 1
    assert count == 1
    output = "\n".join(lines)
    return output

def plug_value_list(vname,vtype,value,check,placeholder):
    value = value.replace("[","").replace("]","")
    if vtype in ["f", "f2", "f3", "c"]:
        if "," in value:
            value = [str(float(k)) for k in value.split(", ")]
        else:
            value = [str(float(k)) for k in value.split(" ") if len(k) > 0]
        for i, m in enumerate(vname):
            if check[i]:
                placeholder = placeholder.replace(m, str(value[i]) + 'f')
    elif vtype == "b":
        assert value in ["true", "false"]
        placeholder = placeholder.replace(vname, value)
    else:
        raise Exception("Unknow type %s" % (vtype))
    return placeholder

def plug_value(vname,vtype,value,placeholder):
    value = value.replace("[","").replace("]","")
    if vtype in ["f", "f2", "f3", "c"]:
        if "," in value:
            value = [str(float(k)) for k in value.split(", ")]
        else:
            value = [str(float(k)) for k in value.split(" ") if len(k) > 0]
        value_str = str(",".join(value))
    elif vtype == "b":
        bdict = {'0':"false", '1':"true", "true":"true", "false":"false"}
        value = bdict[value]
        #assert value in ["true", "false"]
        value_str = value
    else:
        raise Exception("Unknow type %s" % (vtype))
    placeholder = placeholder.replace(vname, value_str)
    return placeholder

def fill_template(items, tempfile, outfile):
    placeholder = open(tempfile).read()
    for item in items:
        vname, vtype, value = [str(k) for k in item]
        vtype = vtype[0:2]
        if "-" not in vname:
            vname = "%%%s%%" % (vname.upper())
            assert vname in placeholder
            placeholder = plug_value(vname,vtype,value, placeholder)
            assert vname not in placeholder
        elif vtype in ["f", "f2", "f3", "c"]:
            count = int(vtype.replace("f", "f1").replace("c", "c3")[-1])
            vname = ["%%%s_%d%%" % (vname.upper(), j) for j in range(count)]
            check = [m in placeholder for m in vname]
            assert sum(check) in [1, count]
            placeholder = plug_value_list(vname,vtype,value, check, placeholder)
            check = [m in placeholder for m in vname]
            assert sum(check) == 0

    return placeholder

def parse_lines(lines):
    cur = 0
    values = {}
    for ptype in ALLTYPES:
        values[ptype] = []
        for i in range(MAX_TYPE_COUNT[ptype]):
            cur += 1
            values[ptype].append(lines[cur])
    return values

def result_to_mdl(fname):
    allranges = load_norm_ranges("%s/matlist_norm_ranges.txt" % DATA_PATH)
    lines = open(fname).read().splitlines()
    predclassid = int(lines[0])
    predmatname = allmats[predclassid]
    gtmatname = "/".join(fname.split("/")[-3:])
    gtmatname = "_".join(gtmatname.split("_")[0:-2])
    values = parse_lines(lines)
    template = "%s/%s_template_params_default.txt" % (result_folder, predmatname)
    tempfile = "%s/%s_template_placeholder.txt" % (result_folder, predmatname)
    outname = "/".join(fname.split("/")[-4:])[0:-4]
    epoch = outname.split("/")[0]
    samplename = outname.split("/")[-1]
   
    outfile = "%s/predictions/%s.mdl" % (result_folder, samplename)
    if os.path.isfile(template):
        tlines = open(template).read().splitlines()
        cur = {k:0 for k in ALLTYPES}
        pred = []
        for line in tlines:
            name, ptype, val = line.split(";")
            ptype = ptype[0:2]
            count = cur[ptype]
            newval = values[ptype][count]
            vmin, vmax = [float(bd) for bd in allranges[predmatname][name]]
            if vmax > 1:
                newval = newval.split(" ")
                newval = [str(float(vl)*vmax) for vl in newval]
                newval = " ".join(newval)
            pred.append([name, ptype, newval])
            cur[ptype] += 1
        output = fill_template(pred, tempfile, outfile)
    else:
        tempfile = "%s/%s_template.mdl" % (result_folder, predmatname)
        output = open(tempfile).read()
    pred_outfolder = "/".join(outfile.split("/")[0:-1])
    if not os.path.isdir(pred_outfolder):
        os.makedirs(pred_outfolder)
    output = replace_mat_name(output, predmatname.split("__")[-1], MYMAT)
    with open(outfile, "w") as f:
        f.write(output)
    return outfile

'''
code to render the prediction. You must have visRTX installed.
'''
def render_prediction(samplepath):

    outfolder = "/cvgl2/u/monaavr/nvidia_mdl_templates"
    command = "/cvgl2/u/lynetcha/programs/VisRTX/build/visRtxRendervMaterials"
    N = 1
    with open("render.sh", "w") as f:
        f.write("#!/usr/bin/env bash\n")
        f.write(f"export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:{pythonpath}\n")
        outfile = result_to_mdl(samplepath)
        mdlpath = outfile.replace(outfolder, "").replace("/", "::")[0:-4]
        mdlname = mdlpath + "::" + MYMAT
        outname = outfile[0:-4]+"_rendering"
        f.write("%s %s %s 1 0\n" % (command, mdlname, outname))
        
    try:
        os.remove(outname + ".ppm")
    except:
        try:
            os.remove(outname + "_lock.txt")
        except:
            print("could not find and delete cached .ppm file")
            pass
    return f"{outname}.ppm"
