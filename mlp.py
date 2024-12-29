import numpy as np

# ⣼⣿⣿⢿⡻⢝⠙⠊⠋⠉⠉⠈⠊⠝⣿⡻⠫⠫⠊⠑⠉⠉⠑⠫⢕⡫⣕⡁⠁
# ⣼⡻⠕⠅⠁⣀⣤⣤⣄⣀⠈⠄⠁⠄⠁⣿⡮⠄⠁⠄⠄⡠⠶⠶⠦⡀⠈⣽⡢
# ⣿⣧⠄⠁⠄⠔⠒⠭⠭⠥⠥⠓⠄⢀⣴⣿⣿⡄⠁⠠⣤⠉⠉⣭⠝⠈⢐⣽⣕
# ⣿⣷⡢⢄⡰⡢⡙⠄⠠⠛⠁⢀⢔⣵⣿⣿⣿⣿⣧⣄⡈⠁⠈⠁⠉⡹⣽⣿⣷
# ⣿⣿⣿⣿⣿⣬⣭⡭⠔⣠⣪⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣵⡒⠫⠿⣿⣿⣿
# ⣿⣿⣿⣿⠿⣛⣥⣶⣿⠟⢁⣶⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⡙⣿⣿⣶⣿⣿⣿
# ⣿⣿⣿⣿⣿⣿⣿⡫⠁⢀⠑⠓⠫⢝⢟⣿⣿⣿⣿⡻⠊⢉⣄⠈⠪⡫⢿⣿⣿
# ⣿⣿⣿⣿⣿⣿⢟⠁⣰⣿⣿⣢⢤⣀⡀⠈⠉⠉⢀⠠⠪⢝⡻⣷⡀⠊⡪⡻⣿
# ⡫⢟⣿⣿⣿⣿⡊⢠⣿⣿⡫⠚⣊⣡⠶⢦⣤⣤⠶⠞⡛⠳⣌⠫⡻⡀⠈⡺⢿
# ⠪⡪⡫⢟⡿⣕⠁⡫⠝⠊⡴⠋⠁⠁⠐⠁⠂⠈⠐⠈⠈⠐⠐⠳⠄⠹⣇⠪⡻
# ⠄⠁⠊⠕⡪⢕⢀⠞⠁⠄⣁⢀⢀⣀⣤⣤⣠⣀⣤⣴⣶⣶⣶⡆⠄⠆⢷⠕⡪
# ⣄⠄⠁⠄⠁⠄⡎⠄⠁⢬⣮⣕⠻⢿⣿⣿⣿⣿⣿⣿⣿⣿⡫⡪⡵⠄⠁⠄⠈
# ⣿⣄⠁⠄⠁⠄⡣⠄⠁⣷⣯⣵⣢⠄⠄⠉⠉⠉⠉⠉⠉⣠⣬⣟⡕⠄⠁⢀⣿
# ⣿⣿⣷⡀⠁⠄⡎⠄⠁⠻⣿⣾⣯⣪⣔⢄⣀⣀⣀⡠⣶⣾⣽⣿⠃⠄⢀⣼⣿


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


class MLP:
    def __init__(self, input_size, hidden_layers, output_size=1, learning_rate=0.01):
        self.input_size = input_size            # input_size = how many features in dataset
        self.hidden_layers = hidden_layers      # list which contains neurons amount in each hidden layer
        # for example [2, 1, 3] means that we want to create 3 hidden layers:
        # first layer - contains 2 neurons
        # second layer - contains 1 neuron
        # third layer - contains 3 neurons
        self.output_size = output_size          # output_size = how many neurons in output layer
                                                # we are predicting one value, so 1 is good choice
        self.learning_rate = learning_rate  # another parameter to adjust

        # weights and biases init
        self.weights = []
        self.biases = []


        # nie chce mi sie po angielsku juz pisac, nie mam c2 nawet
        # dla kazdej warstwy ukrytej robimy sobie macierz wag
        layer_input_size = input_size           # ilosc neuronow w poprzedniej warstwie

        for hidden_size in hidden_layers:
            # wagi na poczatku inicjujemy losowymi wartosciami od -1 do 1
            # macierz wag jest o rozmiarze (ilosc neuronow w poprzedniej warstwie) x (ilosc neuronow w aktualnej warstwie)
            self.weights.append(np.random.uniform(-1, 1, (layer_input_size, hidden_size)))
            # biasy sa po prostu inicjowane jako wektor zawierajacy tyle zer ile jest neuronow w danej warstwie
            self.biases.append(np.zeros((1, hidden_size)))
            layer_input_size = hidden_size
        # inicjalizacja wartwy wyjsciowej
        self.weights.append(np.random.uniform(-1, 1, (layer_input_size, output_size)))
        self.biases.append(np.zeros((1, output_size)))


    def forward(self):
        pass

    def backward(self):
        pass

    def train(self):
        pass

    def predict(self):
        pass
