from PySide6.QtWidgets import (
    QLabel,
    QMainWindow,
    QPushButton,
    QWidget,
    QGridLayout,
    QMessageBox,
    QGroupBox,
    QLineEdit,
    QComboBox,
    QTableWidgetItem,
    QTableWidget,
    QAbstractItemView,
    QHeaderView,
)

from PySide6.QtGui import (
    QGuiApplication,
    QRegularExpressionValidator,
)


from PySide6.QtCore import Qt, QRegularExpression, QUrl
from fileio import read_sample
from backpropagation import train, test
import numpy as np


class Application(QMainWindow):
    def __init__(self):
        super().__init__()
        self.hidden_weights: np.ndarray = np.array([])
        self.output_weights = np.ndarray = np.array([])
        self.train_inputs: np.ndarray = np.array([])
        self.train_classes: np.ndarray = np.array([])
        self.test_inputs: np.ndarray = np.array([])
        self.test_classes: np.ndarray = np.array([])
        self.num_classes: int = 0
        self.num_features: int = 0
        self.num_hidden: int = 5
        self.learning_rate: np.float64 = 1
        self.is_sigmoid: bool = True
        self.max_iterations: int = 200
        self.max_error: np.float64 = 0.0001
        self.stop_by_error: bool = False
        self.conv_matrix: np.ndarray = np.zeros(
            (self.num_hidden, self.num_hidden))

        self.float_validator = QRegularExpressionValidator(
            QRegularExpression(r"[-]?\d*\.?\d*"))
        self.int_validator = QRegularExpressionValidator(
            QRegularExpression(r"[-]?\d*"))

        self.setup()
        self.run()

    def run(self):
        self.show()

    def setup(self):
        self.setWindowTitle("Backpropagation")
        self.show_content()
        self.setFixedSize(850, 280)

        self.center_window()

    def center_window(self):
        frame_geometry = self.frameGeometry()
        center_point = QGuiApplication.primaryScreen().availableGeometry().center()
        frame_geometry.moveCenter(center_point)
        self.move(frame_geometry.topLeft())

    def select_training_file(self):

        try:
            self.train_inputs, self.train_classes, self.num_classes, self.num_features, self.num_hidden = read_sample(
                self)
            self.set_label_training_metadata(
                self.train_inputs.shape[0], self.num_features, self.num_classes)
            self.text_hidden.setText(str(self.num_hidden))

        except Exception:
            QMessageBox.warning(
                self, "Aviso", "As amostras de treinamento n??o foram atualizadas")

    def select_testing_file(self):
        try:
            self.test_inputs, self.test_classes, self.num_classes, self.num_features, self.num_hidden = read_sample(
                self)
            self.set_label_testing_metadata(
                self.test_inputs.shape[0], self.num_features, self.num_classes)

        except Exception:
            QMessageBox.warning(
                self, "Aviso", "As amostras de teste n??o foram atualizadas")

    def set_label_training_metadata(self, num_samples: int, num_features: int, num_classes: int):
        self.label_training_metadata.setText(
            f"Treino: {num_samples} amostras, {num_features} par??metros e {num_classes} classes")

    def set_label_testing_metadata(self, num_samples: int, num_features: int, num_classes: int):
        self.label_testing_metadata.setText(
            f"Teste: {num_samples} amostras, {num_features} par??metros e {num_classes} classes")

    def show_content(self):
        # Divide the grid in 3 visual column blocks
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.grid = QGridLayout(self.central_widget)
        self.grid.setColumnStretch(1, 3)

        self.show_io_section()
        self.show_training_section()
        self.show_testing_section()

    def show_io_section(self):
        # 1st group: input files. Should load two files, one for training and one for testing
        self.input_files_group = QGroupBox("Arquivos de entrada")

        # Buttons to load data
        self.label_training = QLabel("Arquivo para treinamento")
        self.button_training = QPushButton("Selecionar")
        self.button_training.clicked.connect(self.select_training_file)
        self.label_testing = QLabel("Arquivo para teste")
        self.button_testing = QPushButton("Selecionar")
        self.button_testing.clicked.connect(self.select_testing_file)

        # Show metadata about the files
        self.label_training_metadata = QLabel()
        self.set_label_training_metadata(0, 0, 0)

        self.label_testing_metadata = QLabel()
        self.set_label_testing_metadata(0, 0, 0)

        # Add widgets to the group
        self.grid.addWidget(self.input_files_group, 0, 0, 1, 1)
        self.input_files_layout = QGridLayout(self.input_files_group)
        self.input_files_layout.setRowStretch(4, 1)

        self.input_files_layout.addWidget(self.label_training, 0, 0, 1, 1)
        self.input_files_layout.addWidget(self.button_training, 0, 1, 1, 1)
        self.input_files_layout.addWidget(self.label_testing, 1, 0, 1, 1)
        self.input_files_layout.addWidget(self.button_testing, 1, 1, 1, 1)
        self.input_files_layout.addWidget(
            self.label_training_metadata, 2, 0, 1, 1)
        self.input_files_layout.addWidget(
            self.label_testing_metadata, 3, 0, 1, 1)

    def train(self):
        if self.train_inputs.shape[0] == 0:
            QMessageBox.warning(
                self, "Aviso", "Os arquivos de entrada ainda n??o foram carregados")
            return

        self.hidden_weights, self.output_weights = train(
            inputs=self.train_inputs,
            classes=self.train_classes,
            num_classes=self.num_classes,
            num_features=self.num_features,
            num_hidden=self.num_hidden,
            rate=np.float64(self.learning_rate),
            stop_value=self.max_error if self.stop_by_error else self.max_iterations,
            is_sigmoid=self.is_sigmoid,
            stop_by_error=self.stop_by_error,
        )

        QMessageBox.information(
            self, "Aviso", "A rede foi treinada com sucesso")

    def test(self):
        if self.output_weights.shape[0] == 0 or self.hidden_weights.shape[0] == 0:
            QMessageBox.warning(
                self, "Aviso", "A rede ainda n??o foi treinada")
            return

        if self.test_inputs.shape[0] == 0:
            QMessageBox.warning(
                self, "Aviso", "Os arquivos de entrada ainda n??o foram carregados")
            return

        self.conv_matrix = test(
            inputs=self.test_inputs,
            classes=self.test_classes,
            num_classes=self.num_classes,
            is_sigmoid=self.is_sigmoid,
            hidden_weight=self.hidden_weights,
            output_weight=self.output_weights,
        )
        self.update_table(self.conv_matrix)

    def show_training_section(self):
        # 2nd group: training parameters
        self.training_group = QGroupBox("Par??metros de treinamento")

        # Buttons
        self.label_learning_rate = QLabel("Taxa de aprendizado")
        self.text_learning_rate = QLineEdit(str(self.learning_rate))
        self.text_learning_rate.setValidator(self.float_validator)
        self.text_learning_rate.textChanged.connect(self.set_learning_rate)

        self.label_hidden = QLabel("Neur??nios ocultos")
        self.text_hidden = QLineEdit(
            str(self.num_hidden if self.num_hidden else 1))
        self.text_hidden.setValidator(self.int_validator)
        self.text_hidden.textChanged.connect(self.set_num_hidden)

        self.label_activation = QLabel("Fun????o de ativa????o")
        self.combo_activation = QComboBox()
        self.combo_activation.addItems(["Log??stica", "Tg Hiperb??lica"])
        self.combo_activation.currentIndexChanged.connect(self.set_activation)

        self.label_stop = QLabel("Crit??rio de parada")
        self.combo_stop = QComboBox()
        self.combo_stop.addItems(["Erro", "Itera????es"])

        self.combo_stop.setCurrentIndex(0 if self.stop_by_error else 1)
        self.combo_stop.currentIndexChanged.connect(self.set_stop)

        self.label_max_error = QLabel("Erro m??ximo")
        self.text_max_error = QLineEdit(str(self.max_error))
        self.text_max_error.setValidator(self.float_validator)
        self.text_max_error.textChanged.connect(self.set_max_error)

        self.label_max_iterations = QLabel("Itera????es m??ximas")
        self.text_max_iterations = QLineEdit(str(self.max_iterations))
        self.text_max_iterations.setValidator(self.int_validator)
        self.text_max_iterations.textChanged.connect(self.set_max_iterations)

        self.button_train = QPushButton("Treinar")
        self.button_train.clicked.connect(self.train)

        # Layout
        self.grid.addWidget(self.training_group, 0, 1, 1, 1)
        self.training_layout = QGridLayout(self.training_group)

        self.training_layout.addWidget(self.label_learning_rate, 0, 0, 1, 1)
        self.training_layout.addWidget(self.text_learning_rate, 0, 1, 1, 1)
        self.training_layout.addWidget(self.label_hidden, 1, 0, 1, 1)
        self.training_layout.addWidget(self.text_hidden, 1, 1, 1, 1)
        self.training_layout.addWidget(self.label_activation, 2, 0, 1, 1)
        self.training_layout.addWidget(self.combo_activation, 2, 1, 1, 1)
        self.training_layout.addWidget(self.label_stop, 3, 0, 1, 1)
        self.training_layout.addWidget(self.combo_stop, 3, 1, 1, 1)
        self.training_layout.addWidget(self.label_max_error, 4, 0, 1, 1)
        self.training_layout.addWidget(self.text_max_error, 4, 1, 1, 1)
        self.training_layout.addWidget(self.label_max_iterations, 4, 0, 1, 1)
        self.training_layout.addWidget(self.text_max_iterations, 4, 1, 1, 1)
        self.training_layout.addWidget(self.button_train, 5, 0, 1, 2)

        self.training_layout.setRowStretch(6, 1)
        self.switch_stop_condition()

    def switch_stop_condition(self):
        if self.stop_by_error:
            self.label_max_error.show()
            self.text_max_error.show()
            self.label_max_iterations.hide()
            self.text_max_iterations.hide()
        else:
            self.label_max_error.hide()
            self.text_max_error.hide()
            self.label_max_iterations.show()
            self.text_max_iterations.show()

    def set_learning_rate(self, event):
        try:
            number: float = float(event)
            self.learning_rate = number
        except ValueError:
            pass

    def set_num_hidden(self, event):
        try:
            number: int = int(event)
            self.num_hidden = number
        except ValueError:
            pass

    def set_activation(self, event):
        self.is_sigmoid = event == 0

    def set_stop(self, event):
        self.stop_by_error = event == 0
        self.switch_stop_condition()

    def set_max_error(self, event):
        try:
            number: float = float(event)
            self.max_error = number
        except ValueError:
            pass

    def set_max_iterations(self, event):
        try:
            number: int = int(event)
            self.max_iterations = number
        except ValueError:
            pass

    def show_testing_section(self):
        # 3rd group: testing parameters
        self.testing_group = QGroupBox("Teste da amostra")

        # Buttons
        self.button_test = QPushButton("Testar")
        self.button_test.clicked.connect(self.test)

        self.table_conv = QTableWidget(5, 5)
        self.table_conv.setEditTriggers(QAbstractItemView.NoEditTriggers)
        # self.table_conv.setShowGrid(False)
        self.table_conv.horizontalHeader().hide()
        self.table_conv.verticalHeader().hide()
        self.table_conv.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch)
        self.table_conv.verticalHeader().setSectionResizeMode(
            QHeaderView.Stretch)
        self.table_conv.setFixedSize(200, 200)
        for i in range(5):
            for j in range(5):
                if self.table_conv.item(i, j) is None:
                    self.table_conv.setItem(
                        i, j, QTableWidgetItem("0"))

        self.update_table(self.conv_matrix)

    #     # Layout
        self.grid.addWidget(self.testing_group, 0, 2, 1, 1)
        self.testing_layout = QGridLayout(self.testing_group)
        self.testing_layout.addWidget(self.table_conv, 0, 0, 1, 2)
        self.testing_layout.addWidget(self.button_test, 1, 0, 1, 2)

    def update_table(self, new_matrix: np.ndarray):
        for i in range(5):
            for j in range(5):
                item = str(int(new_matrix[i, j]))
                self.table_conv.item(i, j).setText(item)
                self.table_conv.item(i, j).setTextAlignment(Qt.AlignCenter)
