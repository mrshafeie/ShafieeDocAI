from __future__ import annotations

from pathlib import Path

import torch

from shafieedocai.training.line_dataset import PersianLineDataset
from shafieedocai.training.ocr_model import PersianOCRModel
from shafieedocai.training.train import collate_lines, greedy_decode
from shafieedocai.training.vocabulary import CharacterVocabulary


ROOT = Path(__file__).resolve().parent

DATASET_PATH = ROOT / "data" / "synthetic_v3" / "dataset.jsonl"
CHECKPOINT_PATH = (
    ROOT / "models" / "ocr_baseline_rtl_continue" / "best.pt"
)

DEVICE = torch.device("cpu")
STEPS = 600
LEARNING_RATE = 0.001


def minimum_ctc_steps(target: list[int]) -> int:
    repeated = sum(
        target[i] == target[i - 1]
        for i in range(1, len(target))
    )
    return len(target) + repeated


def select_samples(dataset, vocabulary):
    # Try different types of text before filling remaining slots.
    preferred_prefixes = [
        "فاکتور فروش",
        "شماره فاکتور:",
        "فروشنده:",
        "خریدار:",
    ]

    selected = []
    seen_texts = set()

    def try_add(index):
        text = dataset.samples[index]["text"]

        if text in seen_texts:
            return False

        encoded = vocabulary.encode(text)
        sample = dataset[index]
        width = int(sample["image"].shape[-1])

        if width // 4 < minimum_ctc_steps(encoded):
            return False

        selected.append(sample)
        seen_texts.add(text)
        return True

    for prefix in preferred_prefixes:
        for index, metadata in enumerate(dataset.samples):
            if metadata["text"].startswith(prefix):
                if try_add(index):
                    break

        if len(selected) >= 4:
            return selected

    # Fall back to other distinct and CTC-valid samples.
    candidates = sorted(
        range(len(dataset)),
        key=lambda index: (
            len(dataset.samples[index]["text"]),
            index,
        ),
    )

    for index in candidates:
        try_add(index)

        if len(selected) >= 4:
            break

    if len(selected) < 4:
        raise RuntimeError(
            f"Only {len(selected)} suitable unique samples found."
        )

    return selected


def calculate_loss(model, batch, criterion):
    images = batch["images"].to(DEVICE)
    targets = batch["targets"].to(DEVICE)
    input_lengths = batch["input_lengths"].to(DEVICE)
    target_lengths = batch["target_lengths"].to(DEVICE)

    # PersianOCRModel.forward already returns log probabilities.
    log_probs = model(images)

    return criterion(
        log_probs.transpose(0, 1).contiguous(),
        targets,
        input_lengths,
        target_lengths,
    )


def show_predictions(model, batch, vocabulary, title):
    model.eval()

    with torch.inference_mode():
        predictions = greedy_decode(
            model(batch["images"].to(DEVICE)),
            vocabulary,
            batch["input_lengths"],
        )

    print(f"\n{title}")

    for actual, prediction in zip(batch["texts"], predictions):
        print(f"Actual:     {actual}")
        print(f"Prediction: {prediction!r}")
        print(f"Exact match: {'YES' if actual == prediction else 'NO'}")
        print("-" * 35)

    model.train()


def main():
    torch.manual_seed(42)
    torch.set_num_threads(2)

    if not DATASET_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    print(f"Loaded checkpoint epoch: {checkpoint['epoch']}")

    vocabulary = CharacterVocabulary(checkpoint["vocabulary"])

    model = PersianOCRModel(
        num_classes=checkpoint["num_classes"],
        hidden_size=checkpoint["hidden_size"],
        lstm_layers=checkpoint["lstm_layers"],
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state"])

    dataset = PersianLineDataset(
        DATASET_PATH,
        split="train",
        max_width=checkpoint.get("max_width", 512),
        pad_to_max_width=False,
    )

    samples = select_samples(dataset, vocabulary)
    batch = collate_lines(samples, vocabulary)

    print("\nSelected four samples:")
    for text in batch["texts"]:
        print(f"  {text}")

    criterion = torch.nn.CTCLoss(
        blank=CharacterVocabulary.BLANK_ID,
        zero_infinity=False,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.0,
    )

    model.eval()
    with torch.no_grad():
        initial_loss = calculate_loss(model, batch, criterion).item()

    print(f"\nInitial loss: {initial_loss:.5f}")
    show_predictions(
        model, batch, vocabulary, "Predictions before the test"
    )

    model.train()

    for step in range(1, STEPS + 1):
        optimizer.zero_grad(set_to_none=True)

        loss = calculate_loss(model, batch, criterion)

        if not torch.isfinite(loss).item():
            raise RuntimeError(
                f"Non-finite CTC loss at step {step}: {loss.item()}"
            )

        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0,
        )

        if not torch.isfinite(grad_norm).item():
            raise RuntimeError(
                f"Non-finite gradient norm at step {step}"
            )

        optimizer.step()

        if step % 50 == 0:
            print(f"Step {step:03d}/{STEPS} | loss={loss.item():.5f}")

    model.eval()
    with torch.no_grad():
        final_loss = calculate_loss(model, batch, criterion).item()

    print("\n" + "=" * 50)
    print(f"Initial loss: {initial_loss:.5f}")
    print(f"Final loss:   {final_loss:.5f}")

    if initial_loss > 0:
        reduction = (initial_loss - final_loss) / initial_loss * 100
        print(f"Loss reduction: {reduction:.2f}%")

    show_predictions(
        model, batch, vocabulary, "Predictions after 600 steps"
    )

    print("\nTest finished.")
    print("The original best.pt checkpoint was not overwritten.")


if __name__ == "__main__":
    main()