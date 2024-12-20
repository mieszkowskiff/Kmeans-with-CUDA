import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    name = "big_data"
    df = pd.read_csv(f"./ploting/{name}.csv", sep=";")
    df = df[df["iterations"] > 20]
    plt.scatter(df["N"], df["gpu_time"], color="red", label="GPU time")
    plt.scatter(df["N"], df["cpu_time"], color="blue", label="CPU time")
    plt.scatter(df["N"], df["create_data_time"], color="green", label="data creation time")
    plt.legend()
    plt.xlabel("N - number of points in each class")
    plt.ylabel("Time [s]")
    plt.title("Comparison of K-means algorithm on GPU and CPU, \n for number of iterations > 20")   
    plt.savefig(f"./ploting/plots/{name}.png")
    plt.show()