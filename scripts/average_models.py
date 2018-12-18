#!/usr/bin/env python
# -*-coding:utf-8-*-
# Author: alphadl
# Email: liangding.liam@gmail.com
# average_models.py 18/12/18 20:35

import argparse
import torch

def average_models(model_files):

    for i,model_file in enumerate(model_files):
        m = torch.load(model_file)
        if i == 0 :
            avg_model = m
        else :
            for (k,v) in avg_model.items():
                avg_model[k].mul_(i).add_(m[k]).div_(i+1)

    final = avg_model
    return final

def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-models", "-m", nargs="+", required=True,
                        help="List of models")
    parser.add_argument("-output", "-o", required=True,
                        help="Output file")
    opt = parser.parse_args()

    final = average_models(opt.models)
    torch.save(final, opt.output)

if __name__ == "__main__":
    main()
