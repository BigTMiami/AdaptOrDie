ROBERTA_CLASSIFIER_SMALL = {
    "LEARNING_RATE": 2e-5,
    "DROPOUT": 0.1,
    "ENCODER": "CLS",
    "NUM_FEEDFORWARD_LAYERS": 1,
    "FEEDFORWARD_WIDTH_MULTIPLIER": 1,
    "EMBEDDING": "ROBERTA",
    "NUM_EPOCHS": 3, #10
    "PATIENCE": 3,
    "LR_SCHEDULE": 0,
    "GRAD_ACC_BATCH_SIZE": 16,
    "BATCH_SIZE": 16,
}

ROBERTA_CLASSIFIER_MINI = {
    "LEARNING_RATE": 2e-5,
    "DROPOUT": 0.1,
    "ENCODER": "CLS",
    "NUM_FEEDFORWARD_LAYERS": 1,
    "FEEDFORWARD_WIDTH_MULTIPLIER": 1,
    "EMBEDDING": "ROBERTA",
    "NUM_EPOCHS": 10,
    "PATIENCE": 3,
    "LR_SCHEDULE": 0,
    "GRAD_ACC_BATCH_SIZE": 8,
    "BATCH_SIZE": 8,
}

ROBERTA_CLASSIFIER_BIG = {
    "LEARNING_RATE": 2e-5,
    "DROPOUT": 0.1,
    "ENCODER": "CLS",
    "NUM_FEEDFORWARD_LAYERS": 1,
    "FEEDFORWARD_WIDTH_MULTIPLIER": 1,
    "EMBEDDING": "ROBERTA",
    "NUM_EPOCHS": 3,
    "PATIENCE": 3,
    "LR_SCHEDULE": 1,
    "GRAD_ACC_BATCH_SIZE": 16,
    "BATCH_SIZE": 16,
}

HYPERPARAMETERS = {
    "ROBERTA_CLASSIFIER_SMALL": ROBERTA_CLASSIFIER_SMALL,
    "ROBERTA_CLASSIFIER_MINI": ROBERTA_CLASSIFIER_MINI,
    "ROBERTA_CLASSIFIER_BIG": ROBERTA_CLASSIFIER_BIG
}
