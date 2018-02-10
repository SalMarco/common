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

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__file__)

argparser = argparse.ArgumentParser(add_help=True)
argparser.add_argument('-i','--inFile',type=str,help=(" CSV with the data to be analized"),required=True)
argparser.add_argument('-n','--numClus',type=int,help=("Number of cluster [default = 5]"),default=5)
argparser.add_argument('-s','--makeSil',help=("Make na study in n_cluster using Silhouette"),action='store_true')
argparser.add_argument('-o','--outFile',type=str,help=("File with the list of published urls and publish date [OPTIONAL, default= data/O2O_PUBLISHED_URLS.csv ]"),default='data/testKMeansResults.csv')

COL2CONV=['connessa_pc_dedicato','presidiata_maggioranza_fedeli','presidiata_conosce_molto_clientela','presidiata_edicolante','spazio_esclusiva']
# COL2CONV=['connessa_pc_dedicato','presidiata_maggioranza_fedeli','presidiata_conosce_molto_clientela','spazio_esclusiva']

range_n_clusters = [2, 3, 4, 5, 6]

class SearchClusters:
    def __init__(self,**kargs):
        self.inFile = kargs['fi']
        self.outFile = kargs['fo']
        self.numClus = kargs['nk']

    def loadFile(self):
        self.df_orig = pd.read_csv(self.inFile,usecols=['protocollo','diffusa_trasporti','diffusa_ospedale','diffusa_centro_commerciale','diffusa_scuole','diffusa_altro_luogo',
        'diffusa_forte_passaggio','connessa_sito_internet','connessa_comunica_cliente','connessa_comunica_dl','connessa_wifi','connessa_pc_dedicato','presidiata_pos_lottomatica','presidiata_gioco','presidiata_biglietti','presidiata_figurine_flowpack','presidiata_extraeditoriali','presidiata_barscanner','presidiata_maggioranza_fedeli',
        'presidiata_specializzata','presidiata_attivo_con_clientela','presidiata_conosce_molto_clientela','presidiata_edicolante','spazio_locandine','spazio_tipologia','spazio_esclusiva','spazio_parcheggio'],dtype={'connessa_pc_dedicato':'category','presidiata_maggioranza_fedeli':'category','presidiata_conosce_molto_clientela':'category',"presidiata_edicolante":'category','spazio_tipologia':'category',"spazio_esclusiva":'category'})
        self.df_orig['spazio_tipologia'].replace(['NEGOZIO','Negozio'],False,inplace=True)
        self.df_orig['spazio_tipologia'].replace(['CHIOSCO','Chiosco muratura'],True,inplace=True)
        self.df = self.df_orig.copy()
        print(self.df.dtypes)
        for col in COL2CONV:
            self.df[col] = self.df[col].cat.codes
        self.df.drop(['protocollo','presidiata_edicolante'],axis=1,inplace=True)

    def computeKMeans(self):
        kmeans = KMeans(n_clusters=self.numClus,n_init=1000)
        kmeans = DBSCAN()
        kmeans.fit(self.df)
        self.df_orig['cluster'] = kmeans.labels_
        logger.info('DISTRIBUTION OF THE CLUSTERS')
        print(kmeans.labels_)
        logger.info('WRITING DF WITH CLUSTER IN %s',self.outFile)
        self.df_orig.to_csv(self.outFile)

    def calculateSiluettes(self):
        # X = self.df[['spazio_tipologia','presidiata_pos_lottomatica','presidiata_gioco','connessa_pc_dedicato','connessa_comunica_dl',
        # 'connessa_pc_dedicato','presidiata_barscanner','presidiata_biglietti','presidiata_maggioranza_fedeli','presidiata_extraeditoriali']]
        X= self.df
        for n_clusters in range_n_clusters:
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.set_size_inches(18, 7)
            ax1.set_xlim([-0.1, 1])
            ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])
            clusterer = KMeans(n_clusters=n_clusters,n_init=1000)
            # cluster_labels = clusterer.fit_predict(X)
            clusterer.fit(X)
            cluster_labels = clusterer.labels_
            silhouette_avg = silhouette_score(X, cluster_labels)
            print("For n_clusters =", n_clusters,
                  "The average silhouette_score is :", silhouette_avg)
            sample_silhouette_values = silhouette_samples(X, cluster_labels)





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
