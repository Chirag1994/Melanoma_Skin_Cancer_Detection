import torch


class Config:
    EPOCHS = 5
    IMG_SIZE = 512
    ES_PATIENCE = 2
    WEIGHT_DECAY = 0.001
    VAL_BATCH_SIZE = 32 * 2
    RANDOM_STATE = 1994
    LEARNING_RATE = 5e-5
    TRAIN_BATCH_SIZE = 32
    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)
    TRAIN_COLS = [
        "image_name",
        "patient_id",
        "sex",
        "age_approx",
        "anatom_site_general_challenge",
        "target",
        "tfrecord",
    ]
    TEST_COLS = [
        "image_name",
        "patient_id",
        "sex",
        "age_approx",
        "anatom_site_general_challenge",
    ]
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    ################ Setting paths to data input ################

    data_2020 = "../input/jpeg-melanoma-512x512/"
    train_folder_2020 = data_2020 + "train/"
    test_folder_2020 = data_2020 + "test/"
    test_csv_path_2020 = data_2020 + "test.csv"
    train_csv_path_2020 = data_2020 + "train.csv"
    submission_csv_path = data_2020 + "sample_submission.csv"
