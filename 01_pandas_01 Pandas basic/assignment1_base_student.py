import pandas as pd

def main():
    file = input()
    func = input()

    data = pd.read_csv(file)

    if func == 'Q1':
        print(data.shape)
    elif func == 'Q2':
        print(data['score'].max())
    elif func == 'Q3':
        print((data['score'] >= 80).sum())
    else:
        print("No Output")