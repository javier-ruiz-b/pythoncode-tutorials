from os import listdir, remove
from os.path import isfile, join
from download_data import download
import glob
import sys


dataDir = "data"
csvFiles = [f for f in listdir(dataDir) if isfile(join(dataDir, f))]
tickers = [f.split("_")[0] for f in csvFiles]
tickers = list(set(tickers)) #unique

for ticker in tickers:
    fileList = glob.glob(join(dataDir, ticker + '*.csv'))
    for filePath in fileList:
        try:
            remove(filePath)
        except:
            print("Error while deleting file: ",
                  filePath, ". :", sys.exc_info())
        download(ticker)


