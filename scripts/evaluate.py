# Contents of /aeolia/aeolia/scripts/evaluate.py

import argparse
from src.training.trainer import Trainer
from src.utils.metrics import calculate_accuracy, calculate_loss
from src.data.loader import DataLoader

def main():
    parser = argparse.ArgumentParser(description='Evaluate the model performance.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model.')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the evaluation dataset.')
    args = parser.parse_args()

    # Initialize DataLoader
    data_loader = DataLoader(data_path=args.data_path)
    evaluation_data = data_loader.load_batch()

    # Initialize Trainer
    trainer = Trainer(model_path=args.model_path)

    # Evaluate the model
    predictions = trainer.evaluate(evaluation_data)
    accuracy = calculate_accuracy(predictions, evaluation_data.labels)
    loss = calculate_loss(predictions, evaluation_data.labels)

    print(f'Accuracy: {accuracy:.4f}')
    print(f'Loss: {loss:.4f}')

if __name__ == '__main__':
    main()