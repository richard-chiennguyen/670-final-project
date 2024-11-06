import matplotlib.pyplot as plt

def main():
    dataset_sizes = [50, 100, 150, 200]
    
    BL_accuracies = [81.5, 84.2, 84.9, 85.03]
    Masked_BL_accuracies = [86.09, 88.2, 89.91, 90.50]
    proposed_accuracies = [83.53, 85.13, 85.28, 86.5]

    plt.figure(figsize=(10, 6))
    plt.plot(dataset_sizes, BL_accuracies, label="BL", marker="o")
    plt.plot(dataset_sizes, Masked_BL_accuracies, label="Masked-BL", marker="o")
    plt.plot(dataset_sizes, proposed_accuracies, label="Proposed", marker="o")

    plt.title("Accuracy over Dataset Size on Dog vs Cat COCO Dataset")
    plt.xlabel("Dataset Size")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()