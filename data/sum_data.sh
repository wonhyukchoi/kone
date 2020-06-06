#! /usr/bin/env python3
import os
import pandas as pd

if __name__ == "__main__":
    path = os.path.join(os.getcwd(), 'processed')
    files = [os.path.join(path, file_name)
             for file_name in os.listdir(path)
             if file_name.endswith('.csv')]

    acc_df = pd.read_csv(files[0])
    if len(files) > 1:
      for file_name in files[1:]:
          new_df = pd.read_csv(file_name)
          acc_df = acc_df.append(new_df)
    acc_df.drop_duplicates(subset='text', inplace=True)
    acc_df[['text','tag']].to_csv('train_data.csv', encoding='utf-8-sig')
    print("Created train data!")
