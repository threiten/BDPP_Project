import datasets
import zarr
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import remove_comments
import argparse

def clean_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Reset index and remove leftover index column from dataframe
    dataframe: pandas.DataFrame
    """
    dataframe = dataframe.reset_index()
    del dataframe['index']

    return dataframe

def remove_comments_dataset(examples):
    examples['text'] = [remove_comments(ent, ['(', ')']) for ent in examples['text']]
    return examples

def main(options):

    dataset = zarr.load(options.dataFile)
    inpDf = pd.DataFrame({'text': dataset['text'], 'party': dataset['party']})
    trainDf, testDf = train_test_split(inpDf, test_size=0.25, shuffle=True)
    trainDf, valDf = train_test_split(trainDf, test_size=0.1, shuffle=True)

    trainDf = clean_dataframe(trainDf)
    testDf = clean_dataframe(testDf)
    valDf = clean_dataframe(valDf)

    trainDs = datasets.Dataset.from_pandas(trainDf, split="train")
    testDs = datasets.Dataset.from_pandas(testDf, split='test')
    valDs = datasets.Dataset.from_pandas(valDf, split='validation')

    dsDict = datasets.DatasetDict()

    dsDict['train'] = trainDs
    dsDict['test'] = testDs
    dsDict['validation'] = valDs

    dsDict = dsDict.map(remove_comments_dataset, batched=True)
    dsDict.push_to_hub(f"{options.repoName}", token=options.hfToken)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.add_argument_group()
    args.add_argument('--dataFile', '-d', help="Path to the input dataset. Needs to be in zarr format.", action='store', type=str)
    args.add_argument('--hfToken', '-t', help='Token for huggingface hub.', action='store', type=str)
    args.add_argument('--repoName', '-n', help='Name for the dataset repository', action='store', type=str)
    options = parser.parse_args()
    main(options)
