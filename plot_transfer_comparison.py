import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_comparison():
    try:
        # Učitaj podatke
        df_scratch = pd.read_csv("logs/double_pendulum_seed1.csv")
        df_transfer = pd.read_csv("logs/transfer_learning_seed1.csv")
        
        # Dodaj kolonu za tip
        df_scratch['Type'] = 'From Scratch'
        df_transfer['Type'] = 'Transfer Learning'
        
        # Spoji
        df_all = pd.concat([df_scratch, df_transfer])
        
        # Smoothing
        df_all['Smoothed Reward'] = df_all.groupby('Type')['reward'].transform(lambda x: x.rolling(50).mean())
        
        # Crtaj
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df_all, x=df_all.index, y='Smoothed Reward', hue='Type')
        plt.title("Transfer Learning vs. Training from Scratch (Double Pendulum)")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.grid(True, alpha=0.3)
        plt.savefig("logs/transfer_comparison.png")
        print("✓ Grafik sačuvan: logs/transfer_comparison.png")
        plt.show()
        
    except FileNotFoundError:
        print("❌ Nemaš oba fajla. Prvo pokreni main_train_double_pendulum.py i main_transfer_learning.py")

if __name__ == "__main__":
    plot_comparison()
