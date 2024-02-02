from CT_package.Read_CT.CT_dataset import CT_dataset
from CT_package.AIxCT.deep_CT_class import deep_CT


if __name__ == "__main__":

    dataset_train = CT_dataset("/home/edofroses/Train_goldfish/dataset_train/", 5)

    dataset_train.print_info()
    dataset_train.change_values([4, "="], 1)
    dataset_train.split_dataset()
    parametri = {"network" : "SEGNET", "tiles": 400, "batch_size":32, "numero_classi":4, "retrain":"", "num_epochs":150}
    training = deep_CT(dataset_train, parametri)
    dataset_train.print_info()
    training.main_training()
    training.save_info()

    dataset_test = CT_dataset("/home/edofroses/Train_goldfish/dataset_test/", step = 1)
    dataset_test.print_info()
    dataset_test.inizializza_test("/home/edofroses/Train_goldfish/output_dataset_train_02/train_parameters.json")
    dataset_test.print_info()
    testing = deep_CT(dataset_test)
    testing.prediction(mode="3D", subprocess = True)
    dataset_test.print_info()
    dataset_test.mode_filter_25D(filter_size = 4, tipologia = "prediction")
    dataset_test.fill_holes([1,3], tipologia="prediction")
    dataset_test.mode_filter_25D(filter_size = 4, tipologia = "prediction")
    dataset_test.performance()
    dataset_test.print_info()
    dataset_test.save_prediction()