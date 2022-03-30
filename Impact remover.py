from numpy.random import seed
seed(1)
import tensorflow
tensorflow.random.set_seed(5)
import pandas as pd
import numpy as np
from numba import jit
from sklearn import preprocessing
from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras import backend as K


data = pd.read_csv(r"C:\Users\smrko\OneDrive\Desktop\adult-all.csv",header=None, na_values='?')
data = data.dropna()

Adult_data = data.replace(["Male","Female",">50K","<=50K"], [1,0,1,0])

categotical = list(Adult_data.select_dtypes(include = ["object"]).columns)
hot = pd.get_dummies(data = Adult_data, columns=categotical)

Adult = hot.values

count = 5
divide = 5
arr = []
while count <= divide:

    # The ransom seeds (5,
    np.random.seed(5)
    np.random.shuffle(Adult)

    splitt = round(0.8 * len(Adult))
    Adult_train = Adult[:splitt]
    Adult_test = Adult[splitt:]
    # print(hot.columns.get_loc("8_White"))
    # 0 = age, 1 = fnlwgt, educaltion_num = 2, sex = 3, capital_gain = 4, capital_loss = 5, hours_per_week = 6, income = 7
    # standardization
    numerical_train = Adult_train[:, [0, 1, 2, 4, 5, 6]]
    numerical_test = Adult_test[:, [0, 1, 2, 4, 5, 6]]
    numerical_df = pd.DataFrame(data=numerical_train,
                                columns=["age", "fnlwgt", "education_num", "capital_gain", "capital_loss",
                                         "hours_per_week"])
    numerical_df_test = pd.DataFrame(data=numerical_test,
                                     columns=["age", "fnlwgt", "education_num", "capital_gain", "capital_loss",
                                              "hours_per_week"])
    # standardtdization
    standard = numerical_df.columns

    for col in standard:

        col_mean = numerical_df[col].mean()
        col_std = numerical_df[col].std()
        if col_std == 0:
            col_std = 1e-20
        numerical_df[col] = (numerical_df[col] - col_mean) / col_std
        numerical_df_test[col] = (numerical_df_test[col] - col_mean) / col_std
    # numerical data
    train_numerical = numerical_df.values
    test_numerical = numerical_df_test.values

    # categorical data
    categorical_train = Adult_train[:, 8:]
    categorical_test = Adult_test[:, 8:]

    # senitive data
    sensitive_train = Adult_train[:, [3]]
    sensitive_test = Adult_test[:, [3]]
    # target
    target_train = Adult_train[:, [7]]
    target_test = Adult_test[:, [7]]
    # Building the model

    step_1 = np.append(categorical_test, test_numerical, axis=1)
    validation = np.append(step_1, sensitive_test, axis=1)
    first_step = np.append(categorical_train, train_numerical, axis=1)
    predictor = np.append(first_step, sensitive_train, axis=1)
    df = np.append(predictor, target_train, axis=1)
    Dataframe = pd.DataFrame(data=df)

    import aif360
    from aif360.datasets import StructuredDataset
    from aif360.algorithms.preprocessing import DisparateImpactRemover
    import BlackBoxAuditing

    binarydataset = aif360.datasets.BinaryLabelDataset(
        favorable_label=1,
        unfavorable_label=0,
        df=Dataframe,
        label_names=[103],
        protected_attribute_names=[102])

    di = DisparateImpactRemover(repair_level=1)
    dataset_trans = di.fit_transform(binarydataset)
    transformed = dataset_trans.convert_to_dataframe()[0]

    transformed_np = transformed.values
    target_IR = transformed_np[:, [-1]]
    predictor_IR = transformed_np[:, :103]

    # enspired by the datacamp course
    model = Sequential()
    n_cols = predictor_IR.shape[1]
    input_shape = (n_cols,)
    print(n_cols)
    valid_set = (validation, target_test)
    stopping = EarlyStopping(patience=3)

    # layers
    model.add(Dense(100, activation="tanh", input_shape=input_shape))
    model.add(Dense(75, activation="tanh"))
    model.add(Dense(50, activation="tanh"))
    model.add(Dense(25, activation="tanh"))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(optimizer="Adam", loss="binary_crossentropy", metrics=["accuracy"])

    model.fit(predictor_IR, target_IR, epochs=500, batch_size=1000, validation_data=valid_set, callbacks=[stopping])

    scores = model.predict(validation)
    # the data generation phase prepare
    predictions = np.round(scores)
    X = np.append(target_test, predictions, axis=1)
    Y = np.append(sensitive_test, X, axis=1)
    Black = Adult_test[:, [60]]
    white = Adult_test[:, [62]]
    Z = np.append(Black, Y, axis=1)
    Analyse = np.append(white, Z, axis=1)
    Analyse_df = pd.DataFrame(data=Analyse, columns=["white", "black", "sex", "target", "prediction"])

    # all nessassary datapoints
    All = len(Analyse_df.index)
    black_nb = len(Analyse_df[Analyse_df['white'] == 0])
    white_nb = len(Analyse_df[Analyse_df['white'] == 1])
    man_nb = len(Analyse_df[Analyse_df["sex"] == 1])
    woman_nb = len(Analyse_df[Analyse_df["sex"] == 0])
    woman_negative_target = np.where((Analyse_df["sex"] == 0) & (Analyse_df["target"] == 0), True, False)
    WNT_nb = (woman_negative_target == True).sum()
    man_negative_target = np.where((Analyse_df["sex"] == 1) & (Analyse_df["target"] == 0), True, False)
    MNT_nb = (man_negative_target == True).sum()
    woman_positive_target = np.where((Analyse_df["sex"] == 0) & (Analyse_df["target"] == 1), True, False)
    WPT_nb = (woman_positive_target == True).sum()
    man_positive_target = np.where((Analyse_df["sex"] == 1) & (Analyse_df["target"] == 1), True, False)
    MPT_nb = (man_positive_target == True).sum()
    white_negative_target = np.where((Analyse_df["white"] == 1) & (Analyse_df["target"] == 0), True, False)
    WCNT_nb = (white_negative_target == True).sum()
    black_negative_target = np.where((Analyse_df["white"] == 0) & (Analyse_df["target"] == 0), True, False)
    BNT_nb = (black_negative_target == True).sum()
    white_positive_target = np.where((Analyse_df["white"] == 1) & (Analyse_df["target"] == 1), True, False)
    WCPT_nb = (white_positive_target == True).sum()
    black_positive_target = np.where((Analyse_df["white"] == 0) & (Analyse_df["target"] == 1), True, False)
    BPT_nb = (black_positive_target == True).sum()

    Compare_validation_target = np.where(Analyse_df["target"] == Analyse_df["prediction"], True, False)
    right_prediction_all = (Compare_validation_target == True).sum()
    right_black = np.where((Analyse_df["white"] == 0) & (Analyse_df["target"] == Analyse_df["prediction"]), True, False)
    true_black = (right_black == True).sum()
    right_white = np.where((Analyse_df["white"] == 1) & (Analyse_df["target"] == Analyse_df["prediction"]), True, False)
    true_white = (right_white == True).sum()
    right_woman = np.where((Analyse_df["sex"] == 0) & (Analyse_df["target"] == Analyse_df["prediction"]), True, False)
    true_woman = (right_woman == True).sum()
    right_man = np.where((Analyse_df["sex"] == 1) & (Analyse_df["target"] == Analyse_df["prediction"]), True, False)
    true_man = (right_man == True).sum()
    false_postive_man = np.where(
        (Analyse_df["sex"] == 1) & (Analyse_df["target"] == 0) & (Analyse_df["prediction"] == 1), True, False)
    FPM = (false_postive_man == True).sum()
    false_negative_man = np.where(
        (Analyse_df["sex"] == 1) & (Analyse_df["target"] == 1) & (Analyse_df["prediction"] == 0), True, False)
    FNM = (false_negative_man == True).sum()
    false_postive_woman = np.where(
        (Analyse_df["sex"] == 0) & (Analyse_df["target"] == 0) & (Analyse_df["prediction"] == 1), True, False)
    FPW = (false_postive_woman == True).sum()
    false_negative_woman = np.where(
        (Analyse_df["sex"] == 0) & (Analyse_df["target"] == 1) & (Analyse_df["prediction"] == 0), True, False)
    FNW = (false_negative_woman == True).sum()
    false_postive_black = np.where(
        (Analyse_df["white"] == 0) & (Analyse_df["target"] == 0) & (Analyse_df["prediction"] == 1), True, False)
    FPB = (false_postive_black == True).sum()
    false_negative_black = np.where(
        (Analyse_df["white"] == 0) & (Analyse_df["target"] == 1) & (Analyse_df["prediction"] == 0), True, False)
    FNB = (false_negative_black == True).sum()
    false_postive_white = np.where(
        (Analyse_df["white"] == 1) & (Analyse_df["target"] == 0) & (Analyse_df["prediction"] == 1), True, False)
    FPWC = (false_postive_white == True).sum()
    false_negative_white = np.where(
        (Analyse_df["white"] == 1) & (Analyse_df["target"] == 1) & (Analyse_df["prediction"] == 0), True, False)
    FNWC = (false_negative_white == True).sum()

    # the calculation of the analytic numbers

    accrucy = (right_prediction_all / All) * 100
    print("accrucy_all:", accrucy)
    accrucy_woman = (true_woman / woman_nb) * 100
    print("accrucy_woman:", accrucy_woman)
    accrucy_man = (true_man / man_nb) * 100
    print("accrucy_man:", accrucy_man)
    accrucy_white = (true_white / white_nb) * 100
    print("accrucy_white:", accrucy_white)
    accrucy_non_white = (true_black / black_nb) * 100
    print("accrucy_black:", accrucy_non_white)
    FPR_man = (FPM / MNT_nb) * 100
    print("False positive rate man:", FPR_man)
    FPR_woman = (FPW / WNT_nb) * 100
    print("False positive rate woman:", FPR_woman)
    FNR_man = (FNM / MPT_nb) * 100
    print("False negative rate man:", FNR_man)
    FNR_woman = (FNW / WPT_nb) * 100
    print("false negative rate woman:", FNR_woman)
    FPR_White = (FPWC / WCNT_nb) * 100
    print("false positve rate white:", FPR_White)
    FPR_Non_white = (FPB / BNT_nb) * 100
    print("False positive rate Black:", FPR_Non_white)
    FNR_white = (FNWC / WCPT_nb) * 100
    print("False negative rate White:", FNR_white)
    FNR_Non_white = (FNB / BPT_nb) * 100
    print("False negative rate Black:", FNR_Non_white)
    woman_positive_prediction = np.where((Analyse_df["sex"] == 0) & (Analyse_df["prediction"] == 1), True, False)
    WPP_nb = (woman_positive_prediction == True).sum()
    man_positive_prediction = np.where((Analyse_df["sex"] == 1) & (Analyse_df["prediction"] == 1), True, False)
    MPP_nb = (man_positive_prediction == True).sum()
    CV_Score_data = (MPT_nb / man_nb) - (WPT_nb / woman_nb)
    CV_Score_pred = (MPP_nb / man_nb) - (WPP_nb / woman_nb)

    value = [accrucy, accrucy_woman, accrucy_man, accrucy_white, accrucy_non_white, FPR_man, FPR_woman, FNR_man,
             FNR_woman, FPR_White, FPR_Non_white, FNR_white, FNR_Non_white, CV_Score_data, CV_Score_pred]
    # values = np.append(value,value, axis=0)

    arr = np.append(arr, value)

    count = count + 5

header = ["accrucy", "accrucy_woman", "accrucy_man", "accrucy_white", "accrucy_non_white", "FPR_man", "FPR_woman",
          "FNR_man", "FNR_woman", "FPR_White", "FPR_Non_white", "FNR_white", "FNR_Non_white", "CV_Score_data",
          "CV_Score_pred"]
final = np.reshape(arr, (int(divide / 5), int(len(value))))

final_df = pd.DataFrame(final, columns=header)
print(final_df)
from openpyxl import load_workbook

book = load_workbook("C:\\Users\smrko\OneDrive\Desktop\AI_Neural.xlsx")
writer = pd.ExcelWriter("C:\\Users\smrko\OneDrive\Desktop\AI_Neural.xlsx", engine='openpyxl')
writer.book = book

final_df.to_excel(writer, sheet_name='Impact_Remover')
writer.save()
# first opening of the excel file
# writer = pd.ExcelWriter("C:\\Users\smrko\OneDrive\Desktop\AI_Neural.xlsx", engine = 'openpyxl')
# final_df.to_excel(writer, sheet_name = 'x2')
# writer.save()


