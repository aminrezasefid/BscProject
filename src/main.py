from DataTransformer import DataTransformer
from Trainer import *
from GNNModel import GNNModel
from Dataset import Dataset
from utils import save_to_pickle, load_from_pickle
from pathlib import Path
dir_prefix = "../"
outfile = "{}data_{}_model_{}.{}"
pickle_dir = "{}data/models/"
images_dir = "{}data/img/"



def run_gnn_cont(filename,mainCode,f_mode,n_f, dir_prefix="../", lr=0.0001, exp_num=0, **kwargs):
    """
    Training a GNN model using continuous evaluation. The model is then saved into pickle.
    :param filename: the name of the file with the input data
    :param dir_prefix: directory prefix for model saving
    :param lr: a learning rate for training
    :param exp_num: experiment number
    :param kwargs: additional parameters for the model
    :return:
    """
    pickleFileName=Path(filename).stem
    codeName=""
    if mainCode:
        codeName="ThesisCode"
    else:
        codeName=""
    modeName=""
    if f_mode=='variable':
        modeName="variable_features"
    elif f_mode=='fixed':
        modeName="fixed_features"
    featureCount=""
    if n_f!=None:
        featureCount=f"{n_f}feature"
    print(f"acc_{pickleFileName}_{codeName}_{modeName}_{featureCount}.pickle")
    dataset = Dataset(filename=filename)
    data = dataset.process() # load and process all the data
    epochs = [30,100] # number of initial epochs
    test_acc = []
    val_acc = []
    n_all_teams = 0
    n_all_teams = data.n_teams
    model=None
    if mainCode:
        model = GNNModel(n_all_teams,10,runMainCode=mainCode, **kwargs)
    else:
        model=GNNModel(n_all_teams,n_f,runMainCode=mainCode, **kwargs)
    print("GNN model, data {}", 0)
    continuous_evaluation(data, model,f_mode,n_f, epochs[0],lr=lr, batch_size=9)
    test_cont(data, model, data.data_test, "test")
    print("accuracy on testing data is: {}".format(data.test_accuracy))
    file = outfile.format(pickle_dir.format(dir_prefix), 0, exp_num, "pickle")
    data_to_save = {"data": data, "model": model, "epochs": epochs}
    data_to_save2 = {"val_accuracy": data.val_accuracy , "train_accuracy":data.train_accuracy,"test_accuracy":data.test_accuracy,"baseline":data.baseline}
    save_to_pickle(f"acc_{pickleFileName}_{codeName}_{modeName}_{featureCount}.pickle", data_to_save2)
    save_to_pickle(file, data_to_save)
    test_acc.append(data.test_accuracy)
    val_acc.append(data.val_acc)

    test_accuracy = sum(test_acc) / len(test_acc)
    val_accuracy = sum(val_acc) / len(val_acc)
    file = outfile.format(pickle_dir.format(dir_prefix), "all", exp_num, "pickle")
    data_to_save = {"test_acc": test_acc, "val_acc": val_acc}
    save_to_pickle(file, data_to_save)
    return test_accuracy, val_accuracy


def run_exist_model(model_file: str, lr:float = 0.0001):
    """
    Run a trained model on the testing data and retrain the model using it (sliding testing set)
    :param model_file: input file with the saved model and the data
    :param lr: tuple of learning rates for training and validation
    :return:
    """
    m = load_from_pickle(model_file)
    data = m["data"]
    model = m["model"]
    epochs = [60]
    model.eval()
    test_cont(data, model, data.data_test, "test")
    print("accuracy on testing data is: {}".format(data.test_accuracy))
    continuous_evaluation(data, model, epochs[0], lr=lr, batch_size=9, mode="test")



# _________________________Unused functions_________________________






if __name__ == '__main__':
    # 5: gnn cont, 8: flat cont, 11: run_exist, 9: vis cont, 10: vis embedding,
    # 12: confusion matrix, 13: rps
    # UNUSED:
    # 0:Flat, 1:PageRank, 2: GNN, 3: visualization

    function_id = 5
    exp_num = "0"
    dataset_filename = "../data/soccer_all_leagues.csv"
    dataset_filename = "../data/NHL.csv"
    dataset_filename = "../data/GER1_all.csv"
    dataset_filename= "../data/Primier2010-2020.csv"
    #dataset_filename= "../data/Primier2018-2020.csv"
    dataset_filename = "../data/Bundesliga2010-2020.csv"
    #dataset_filename = "../data/Bundesliga2018-2020.csv"
    #dataset_filename = "../data/Fake.csv"
    model_filename = "../data_0_model_154.pickle"

    if function_id == 5:
        run_gnn_cont(dataset_filename,mainCode=True,f_mode=None,n_f=None)

