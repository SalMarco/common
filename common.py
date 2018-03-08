import os, sys
from datetime import datetime
import pandas as pd
import logging
from sklearn.cluster import KMeans, DBSCAN
import argparse
import numpy as np
from sklearn.metrics import silhouette_samples, silhouette_score

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import magic

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__file__)

argparser = argparse.ArgumentParser(add_help=True)
argparser.add_argument('-i','--inFile',type=str,help=(" CSV with the data to be analized"),required=True)
argparser.add_argument('-n','--numClus',type=int,help=("Number of cluster [default = 5]"),default=5)
argparser.add_argument('-s','--makeSil',help=("Make na study in n_cluster using Silhouette"),action='store_true')
argparser.add_argument('-o','--outFile',type=str,help=("File with the list of published urls and publish date [OPTIONAL, default= data/O2O_PUBLISHED_URLS.csv ]"),default='data/testKMeansResults.csv')


########################
###ARGPARSER CON UNO O L'ALTRO
###############################
argparser = argparse.ArgumentParser(add_help=True)
argparser.add_argument('-o','--outFile',type=str,help=("Outfile"))
group = argparser.add_mutually_exclusive_group(required=True)
group.add_argument('-c','--csv',type=str,help=(" CSV with the data to be analized"))
group.add_argument('-z','--zip',type=str,help=(".zip with the data to be analized"))


def main():
    start=datetime.now()
    logger.info("START")
    args = argparser.parse_args()
    inFile = args.inFile
    outFile = args.outFile
    numClus = args.numClus
    makeSil = args.makeSil
    logger.info("RUNNING WITH INFILE %s, OUTFILE %s AND NUM OF CLUSTER %d",inFile,outFile,numClus)
    sc = SearchClusters(fi=inFile,fo=outFile,nk=numClus)
    logger.info('READING THE CSV')
    sc.loadFile()
    logger.info('COMPUTING KMENAS')
    if makeSil:
        sc.calculateSiluettes()
    else:
        sc.computeKMeans()
    logger.info('DONE IN %s',str(datetime.now()-start))


if __name__ == "__main__":
    main()

def outFileName(**kargs):
    if kargs['fo']:
        ret=kargs['fo']
        path=os.path.dirname(kargs["fo"])
        name=os.path.basename(kargs["fo"])
        sp=name.split(".")
        outname_json=('%s.json'%sp[0])
        outname_report=('%s_report.%s'%(sp[0],sp[1]))
        return ret, os.path.join(path,outname_json), os.path.join(path,outname_report)
    else:
        path=os.path.dirname(kargs["fi"])
        outname="%s_%s.csv"%(DEF_OUTNAME,kargs['et'])
        outname_report="%s_%s_report.csv"%(DEF_OUTNAME,kargs['et'])
        outname_json="%s_%s.json"%(DEF_OUTNAME,kargs['et'])
        return os.path.join(path,outname), os.path.join(path,outname_json), os.path.join(path,outname_report)


#Per fare  il check del tipo di file e aprirlo in ogni caso
def checkFileType(self,inFile,ext):
        file_type=magic.from_file(inFile,mime=True)
        if file_type=="application/gzip":
            fi = gzip.open(inFile,'%st'%ext)
        else:
            fi = open(inFile,ext)
        logger.info('FILE TYPE FOR %s IS %s',inFile,file_type)
        return fi
