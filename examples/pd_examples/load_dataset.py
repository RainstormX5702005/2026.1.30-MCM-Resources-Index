import kagglehub


def main():
    mypath = "d:/Projects/Coursework/math_modeling/data/iris"
    path = kagglehub.dataset_download("uciml/iris", path=mypath)
    print(f"Dataset downloaded to: {path}")


if __name__ == "__main__":
    main()
