if __name__ == "__main__":
    import pandas as pd

    # Load dataset
    df = pd.read_excel("elf_dataset.xlsx")   # adjust path if needed
    target_col = "DEMAND"

    # Train + Save
    model, scaler_X, scaler_y, history, results = train_lstm_model(df, target_col, save_dir=".")
    print("✅ Model trained and saved successfully!")
    print("📊 Evaluation Results:", results)
