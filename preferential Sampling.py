from numpy.random import seed
seed(1)
import tensorflow
tensorflow.random.set_seed(5)
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
import itertools

data = pd.read_csv(r"C:\Users\smrko\OneDrive\Desktop\adult-all.csv",header=None, na_values='?')
data = data.dropna()

Adult_data = data.replace(["Male","Female",">50K","<=50K"], [1,0,1,0])

categotical = list(Adult_data.select_dtypes(include = ["object"]).columns)
hot = pd.get_dummies(data = Adult_data, columns=categotical)

Adult = hot.values

count = 5
divide = 500
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
    target_train = Adult_train[:, 7]
    target_test = Adult_test[:, [7]]
    # Building the model
    from tensorflow import keras
    from keras.layers import Dense
    from keras.models import Sequential
    from keras.callbacks import EarlyStopping
    from keras import backend as K

    step_1 = np.append(categorical_test, test_numerical, axis=1)
    validation = np.append(step_1, sensitive_test, axis=1)
    first_step = np.append(categorical_train, train_numerical, axis=1)
    predictor = np.append(first_step, sensitive_train, axis=1)
    df_predictor = pd.DataFrame(data=predictor)

    Sample_structure = pd.DataFrame({'group': [1, 1, 0, 0], 'class': [1, 0, 1, 0]})

    # using nave bayers to rank the data
    R = GaussianNB()
    NB_prop = R.fit(predictor, target_train).predict_proba(predictor)

    df_predictor['class'] = target_train
    df_predictor['prob'] = [record[1] for record in NB_prop]
    # weighting the data to make the equalize expected adn real representation
    weights = [[(len(df_predictor[df_predictor[102] == s]) / (len(df_predictor)) * len(
        df_predictor[df_predictor['class'] == c]) / len(df_predictor))
                / (len(df_predictor[(df_predictor[102] == s) & (df_predictor['class'] == c)]) / len(df_predictor))
                for c in [1, 0]] for s in [1, 0]]

    sizes = [[len(df_predictor[(df_predictor[102] == s) & (df_predictor['class'] == c)]) for c in [1, 0]] for s in
             [1, 0]]

    Sample_structure['weight'] = [i for j in weights for i in j]
    Sample_structure['size'] = [i for j in sizes for i in j]
    # i choose here a parameter between 0.6; 0.7; 0.8; 0.9 and choose the best result
    Sample_structure = Sample_structure.assign(
        num=abs(round(0.6 * (Sample_structure["size"] - Sample_structure["weight"] * Sample_structure["size"]))))

    # sorting the data for sampling i changed it so the data the farest away from the border is deleted this data seems to have high propability to be outlayers
    # will test both versions my idea and the idea from the paper and choose the best
    woman_desired = df_predictor[(df_predictor[102] == 0) & (df_predictor['class'] == 1)]
    woman_desired = woman_desired.sort_values(by='prob', ascending=True)
    woman_undesired = df_predictor[(df_predictor[102] == 0) & (df_predictor['class'] != 1)]
    woman_undesired = woman_undesired.sort_values(by='prob', ascending=False)
    man_desired = df_predictor[(df_predictor[102] != 0) & (df_predictor['class'] == 1)]
    man_desired = man_desired.sort_values(by='prob', ascending=True)
    man_undesired = df_predictor[(df_predictor[102] != 0) & (df_predictor['class'] != 1)]
    man_undesired = man_undesired.sort_values(by='prob', ascending=False)
    print(woman_undesired)
    woman_desired_dub = pd.concat([woman_desired] * 3, ignore_index=True)
    man_undesired_dub = pd.concat([man_undesired] * 3, ignore_index=True)

    np_Sample_structure = Sample_structure.values
    splitt_man_desired = round(np_Sample_structure[0, 4])
    splitt_man_undesired = round(np_Sample_structure[1, 4])
    splitt_woman_desired = round(np_Sample_structure[2, 4])
    splitt_woman_undesired = round(np_Sample_structure[3, 4])

    woman_desired_copies = woman_desired_dub[:(splitt_woman_desired)]
    man_undesired_copies = man_undesired_dub[:(splitt_man_undesired)]
    man_desired_drop = man_desired.drop(man_desired.index[range(splitt_man_desired)])
    woman_undesired_drop = woman_undesired.drop(woman_undesired.index[range(splitt_woman_undesired)])

    X_prime = pd.concat(
        [woman_desired_copies, man_undesired_copies, man_desired_drop, woman_undesired_drop, man_undesired,
         woman_desired])
    df_prime = X_prime.sample(frac=1)
    df_prime_drop = df_prime.drop(['prob'], axis=1)
    Preferencial_Sampling_trainingset = df_prime_drop.values
    predictor_Preferencial_Sampling = Preferencial_Sampling_trainingset[:, :103]
    target_Preferencial_Sampling = Preferencial_Sampling_trainingset[:, [103]]

    # Building the model
    from tensorflow import keras
    from keras.layers import Dense
    from keras.models import Sequential
    from keras.callbacks import EarlyStopping
    from keras import backend as K

    # enspired by the datacamp course
    model = Sequential()
    n_cols = predictor_Preferencial_Sampling.shape[1]
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

    model.fit(predictor_Preferencial_Sampling, target_Preferencial_Sampling, epochs=500, batch_size=40,
              validation_data=valid_set, callbacks=[stopping])

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

final_df.to_excel(writer, sheet_name='Preferential_Sampling')
writer.save()
# first opening of the excel file
# writer = pd.ExcelWriter("C:\\Users\smrko\OneDrive\Desktop\AI_Neural.xlsx", engine = 'openpyxl')
# final_df.to_excel(writer, sheet_name = 'x2')
# writer.save()















