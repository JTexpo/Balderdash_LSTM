import numpy as np

from lstm_model.lstm import LSTM

class Optimizer:
    def __init__(
        self,
        v_forget_gate_weights: np.array,
        s_forget_gate_weights: np.array,
        v_input_gate_weights: np.array,
        s_input_gate_weights: np.array,
        v_output_gate_weights: np.array,
        s_output_gate_weights: np.array,
        v_gate_gate_weights: np.array,
        s_gate_gate_weights: np.array,
        v_hidden_output_weights: np.array,
        s_hidden_output_weights: np.array,
        beta_1: float,
        beta_2: float,
        learning_rate:float
    ):
        self.v_forget_gate_weights = v_forget_gate_weights
        self.s_forget_gate_weights = s_forget_gate_weights

        self.v_input_gate_weights = v_input_gate_weights
        self.s_input_gate_weights = s_input_gate_weights

        self.v_output_gate_weights = v_output_gate_weights
        self.s_output_gate_weights = s_output_gate_weights

        self.v_gate_gate_weights = v_gate_gate_weights
        self.s_gate_gate_weights = s_gate_gate_weights

        self.v_hidden_output_weights = v_hidden_output_weights
        self.s_hidden_output_weights = s_hidden_output_weights

        self.beta_1 = beta_1
        self.beta_2 = beta_2

        self.learning_rate = learning_rate

def initialize_optimizer(lstm: LSTM, beta_1: float, beta_2: float, learning_rate:float) -> Optimizer:
    return Optimizer(
        v_forget_gate_weights=np.zeros(lstm.forget_gate_weights.shape),
        s_forget_gate_weights=np.zeros(lstm.forget_gate_weights.shape),
        v_input_gate_weights=np.zeros(lstm.input_gate_weights.shape),
        s_input_gate_weights=np.zeros(lstm.input_gate_weights.shape),
        v_output_gate_weights=np.zeros(lstm.output_gate_weights.shape),
        s_output_gate_weights=np.zeros(lstm.output_gate_weights.shape),
        v_gate_gate_weights=np.zeros(lstm.gate_gate_weights.shape),
        s_gate_gate_weights=np.zeros(lstm.gate_gate_weights.shape),
        v_hidden_output_weights=np.zeros(lstm.hidden_output_weights.shape),
        s_hidden_output_weights=np.zeros(lstm.hidden_output_weights.shape),
        beta_1=beta_1,
        beta_2=beta_2,
        learning_rate=learning_rate
    )