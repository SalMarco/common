import luigi
import bg2gcs
import logging
import subprocess
import sys
from datetime import datetime
import pandas as pd
import seasonFFT_SessionPar as season

logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(message)s', level = logging.INFO)
logger = logging.getLogger(__file__)

class ImportDataTask(luigi.Task):
    from_date = luigi.DateParameter()
    to_date = luigi.DateParameter()

    def requires(self):
        pass

    def run(self):

       logger.info('STARTED')
       start = datetime.now()

       logger.info('FROM DATE: %s'% self.from_date.strftime("%Y-%m-%d"))
       logger.info('TO DATE: %s'% self.to_date.strftime("%Y-%m-%d"))

       d = bg2gcs.Downloader(fromdate = self.from_date, todate = self.to_date, urls = '')

       input_df = d.download()

       logger.info('WRITING FILE: data/all_sessions_%s_%s.csv'%(self.from_date.strftime("%Y-%m"), self.to_date.strftime("%Y-%m")))

       with self.output().open('w') as t:
           input_df.to_csv(t, sep = ',')

       logger.info('TIME ELAPSED: %s'% str(datetime.now() - start))

    def output(self):
        return luigi.LocalTarget('data/all_sessions_%s_%s.csv'%(self.from_date.strftime("%Y-%m"), self.to_date.strftime("%Y-%m")))

class RunTbatsR(luigi.Task):
    from_date = luigi.DateParameter()
    to_date = luigi.DateParameter()

    published = luigi.LocalTarget('data/published_url.csv')

    def requires(self):
        return {'sessions': ImportDataTask(self.from_date, self.to_date),
                'published': SelectPublished(self.from_date, self.to_date)
                }

    def run(self):
        with self.output().open('w') as o, self.input()['sessions'].open('r') as i, self.published.open('r') as p:
           subprocess.check_call('Rscript parallel_tbats.R -i %s -o %s -p %s'%(i.name, o.name, p.name), shell=True)

    def output(self):
        return luigi.LocalTarget('data/season_tbat_%s_%s.csv'%(self.from_date.strftime("%Y-%m"), self.to_date.strftime("%Y-%m")))

class SelectPublished(luigi.Task):
    from_date = luigi.DateParameter()
    to_date = luigi.DateParameter()

    published = luigi.LocalTarget('data/published_url.csv')

    def requires(self):
        return ImportDataTask(self.from_date, self.to_date)

    def run(self):

        with self.output().open('w') as o, self.input().open('r') as i, self.published.open('r') as p:
            subprocess.check_call('python selectPublishedUrls.py -i %s -o %s -p %s'%(i.name, o.name, p.name), shell=True)

    def output(self):
        return luigi.LocalTarget('data/published_%s_%s.csv'%(self.from_date.strftime("%Y-%m"), self.to_date.strftime("%Y-%m")))

class RunFFTpy(luigi.Task):
    from_date = luigi.DateParameter()
    to_date = luigi.DateParameter()

    def requires(self):
        return {'sessions': ImportDataTask(self.from_date, self.to_date),
                'dates': SelectPublished(self.from_date, self.to_date)
                }

    def run(self):

        with self.output().open('w') as o, self.input().open('r') as i:
            subprocess.check_call('python seasonFFT_SessionPar.py -i %s -o %s -t %s -f %s -p %s'%(i.name, o.name, self.to_date, self.from_date), shell=True)

    def output(self):
        return luigi.LocalTarget('data/season_fft_%s_%s.csv'%(self.from_date.strftime("%Y-%m"), self.to_date.strftime("%Y-%m")))

class ComputeSeasonality(luigi.Task):
    from_date = luigi.DateParameter()
    to_date = luigi.DateParameter()

    model = luigi.LocalTarget('data/rf_model.Rdata')

    def requires(self):
        return {'fft': RunFFTpy(self.from_date, self.to_date),
                'tbat': RunTbatsR(self.from_date, self.to_date),
                'season': SelectSeason(self.from_date, self.to_date)}

    def run(self):
        with self.output().open('w') as o, self.input()['fft'].open('r') as fft, self.input()['tbat'].open('r') as tbat, self.model.open('r') as m:
           subprocess.check_call('Rscript predict_seasonality.R -f %s -t %s -o %s -m %s'%(fft.name, tbat.name, o.name, m.name), shell=True)

    def output(self):
        return luigi.LocalTarget('data/season_final_%s_%s.csv'%(self.from_date.strftime("%Y-%m"), self.to_date.strftime("%Y-%m")))

class SelectSeason(luigi.Task):
    from_date = luigi.DateParameter()
    to_date = luigi.DateParameter()

    def requires(self):
        return SelectPublished(self.from_date, self.to_date)

    def run(self):
        with self.input().open('r') as i, self.output.open('w') as o:
            subprocess.check_call('python single_season.py -i %s -o %s'%(i.name, o.name), shell = True)

    def output(self):
        return luigi.LocalTarget('data/single_season_%s_%s.csv'%(self.from_date.strftime("%Y-%m"), self.to_date.strftime("%Y-%m")))
