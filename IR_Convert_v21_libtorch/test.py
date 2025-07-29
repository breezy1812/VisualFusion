import pandas as pd
import matplotlib.pyplot as plt

# 讀取CSV
df = pd.read_csv('image_homo_errors.csv')

# 依照 Is_Cubic 分組
cubic_yes = df[df['Is_Cubic'] == 'Yes']
cubic_no = df[df['Is_Cubic'] == 'No']

plt.figure(figsize=(18, 7))
plt.plot(cubic_yes['Image_Name'], cubic_yes['Euclidean_Error'], color='blue', marker='o', label='cubic=Yes')
plt.plot(cubic_no['Image_Name'], cubic_no['Euclidean_Error'], color='orange', marker='o', label='cubic=No')
plt.xlabel('Image Name', fontsize=14)
plt.ylabel('Euclidean Error', fontsize=14)
plt.title('Euclidean Error: Cubic vs Linear Interpolation', fontsize=16)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.legend()
plt.tight_layout()
plt.grid(True, alpha=0.3)
plt.savefig('image_homo_errors_cubic_vs_linear.png', dpi=300, bbox_inches='tight')
plt.close()
print('Line plot saved as image_homo_errors_cubic_vs_linear.png')