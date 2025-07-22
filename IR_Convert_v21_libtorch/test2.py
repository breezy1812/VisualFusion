import pandas as pd
import matplotlib.pyplot as plt

# 讀取CSV
df = pd.read_csv('video_homo_errors.csv')

# 取得所有影片名稱
video_names = df['video_name'].unique()

for video in video_names:
    sub_df = df[df['video_name'] == video]
    cubic_df = sub_df[sub_df['isUsingCubic'] == 'true']
    linear_df = sub_df[sub_df['isUsingCubic'] == 'no']

    plt.figure(figsize=(10, 6))
    if not cubic_df.empty:
        plt.plot(cubic_df['frame'], cubic_df['Err'], marker='o', label='Cubic')
    if not linear_df.empty:
        plt.plot(linear_df['frame'], linear_df['Err'], marker='s', label='Linear')
    plt.title(f'Error vs Frame: {video} (Cubic vs Linear)')
    plt.xlabel('Frame')
    plt.ylabel('Euclidean Error')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{video}_compare.png')
    plt.close()