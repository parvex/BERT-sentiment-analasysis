import os

from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score, recall_score, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader
import torch
from torch import nn

from ReviewDataset import ReviewDataset
from SentimentClassifier import SentimentClassifier
from consts import class_names, RANDOM_SEED, PRE_TRAINED_MODEL_NAME, TOKEN_MAX_LEN, BATCH_SIZE, EPOCHS
from predict_review.predict_review import predict_single_review
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
from sklearn.utils.class_weight import compute_class_weight
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd

from preprocess import Preprocessing
from training import train_epoch, eval_model, get_predictions, show_confusion_matrix
from util import DATASET_PATH, download_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = ReviewDataset(
        reviews=df.reviewText.to_numpy(),
        targets=df.overall.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=4
    )


def plot_history(history):
    plt.plot(history['train_acc'], label='train accuracy')
    plt.plot(history['val_acc'], label='validation accuracy')

    plt.title('Training history')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.ylim([0, 1])


def show_metrics(y_pred, y_pred_probs, y_test):
    cm = confusion_matrix(y_test, y_pred)
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    show_confusion_matrix(df_cm)

    torch.mean(torch.abs(y_test - y_pred) * 1.0)

    y_test_vec = label_binarize(y_test, classes=[0, 1, 2, 3, 4])
    precision, recall, _ = precision_recall_curve(y_test_vec.ravel(), y_pred_probs.ravel())
    average_precision = average_precision_score(y_test_vec, y_pred_probs, average="micro")
    average_recall = recall_score(y_test, y_pred, average="micro")
    f1_score = 2 * average_precision * average_recall / (average_precision + average_recall)
    pr_auc = auc(recall, precision)

    plt.figure()
    plt.step(recall, precision, where='post')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(
        'Average precision, recall, area under curve, f1 score, micro-averaged over all classes: AP={0:0.2f}, AR={1:0.2f}, AUC={2:0.2f}, F1={3:0.2f}'
            .format(average_precision, average_recall, pr_auc, f1_score))


def main():
    if not os.path.exists(DATASET_PATH):
        download_dataset()
    df = pd.read_csv("./data/dataset.csv")
    df['overall'] -= 1

    tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

    df_train, df_test = train_test_split(df, test_size=0.25, random_state=RANDOM_SEED, stratify=df[['overall']])
    train_data_loader = create_data_loader(df_train, tokenizer, TOKEN_MAX_LEN, BATCH_SIZE)
    test_data_loader = create_data_loader(df_test, tokenizer, TOKEN_MAX_LEN, BATCH_SIZE)

    model = SentimentClassifier(len(class_names), PRE_TRAINED_MODEL_NAME)
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
    total_steps = len(train_data_loader) * EPOCHS

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    # class weights for loss function for imbalanced problem
    class_weights = compute_class_weight(classes=[0, 1, 2, 3, 4], y=df_train['overall'], class_weight='balanced')
    class_weights = torch.FloatTensor(class_weights).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights).to(device)

    history = defaultdict(list)
    best_accuracy = 0

    for epoch in range(EPOCHS):
        print(f'Epoch {epoch + 1}/{EPOCHS}')
        print('-' * 10)

        train_acc, train_loss = train_epoch(
          model,
          train_data_loader,
          loss_fn,
          optimizer,
          device,
          scheduler,
          len(df_train)
        )

        print(f'Train loss {train_loss} accuracy {train_acc}')

        val_acc, val_loss = eval_model(
          model,
          test_data_loader,
          loss_fn,
          device,
          len(df_test)
        )

        print(f'Val loss {val_loss} accuracy {val_acc}')
        print()

        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)

        if val_acc > best_accuracy:
            torch.save(model.state_dict(), 'best_model_state.bin')
            best_accuracy = val_acc

    plot_history(history)

    test_acc, _ = eval_model(
        model,
        test_data_loader,
        loss_fn,
        device,
        len(df_test)
    )

    y_review_texts, y_pred, y_pred_probs, y_test = get_predictions(
        model,
        test_data_loader,
        device
    )

    os.makedirs("model", exist_ok=True)
    torch.save(model.state_dict(), "model/model.pt")

    show_metrics(y_pred, y_pred_probs, y_test)

    preprocessing = Preprocessing()
    predict_single_review("I like it, perfect", preprocessing, tokenizer, model, device)


if __name__ == "__main__":
    main()

