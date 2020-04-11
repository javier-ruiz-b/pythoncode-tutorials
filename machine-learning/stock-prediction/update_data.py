from os import listdir, remove
from os.path import isfile, join
from download_data import download
import glob

dataDir = "data"
csvFiles = [f for f in listdir(dataDir) if isfile(join(dataDir, f))]
tickers = [f.split("_")[0] for f in csvFiles]
tickers = list(set(tickers)) #unique

for ticker in tickers:
    fileList = glob.glob(join(dataDir, ticker + '*.csv'))
    for filePath in fileList:
        try:
            os.remove(filePath)
        except as err:
            print("Error while deleting file: ", filePath, ". :", err)
    download(ticker)


