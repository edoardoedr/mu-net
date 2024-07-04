from CT_package.Read_CT.CTDataset import CTDataset
from CT_package.AIxCT.DeepCT import DeepCT


if __name__ == "__main__":

    dataset_train = CTDataset("/../dataset_train/", 5)

    dataset_train.print_info()
    dataset_train.change_values([4, "="], 1)
    dataset_train.split_dataset()
    parametri = {"network" : "SEGNET", "tiles": 400, "batch_size":32, "numero_classi":4, "retrain":"", "num_epochs":150}
    training = DeepCT(dataset_train, parametri)
    dataset_train.print_info()
    training.train_dataset()
    training.save_info()

    dataset_test = CTDataset("/../dataset_test/", step = 1)
    dataset_test.print_info()
    dataset_test.inizializza_test("/../output_dataset_train_02/train_parameters.json")
    dataset_test.print_info()
    testing = DeepCT(dataset_test)
    testing.prediction(mode="3D", subprocess = True)
    dataset_test.print_info()
    dataset_test.mode_filter_25D(filter_size = 4, tipologia = "prediction")
    dataset_test.fill_holes([1,3], tipologia="prediction")
    dataset_test.mode_filter_25D(filter_size = 4, tipologia = "prediction")
    dataset_test.performance()
    dataset_test.print_info()
    dataset_test.save_prediction()