"""Core models for our experiments."""

from typing import Optional

import torch
import torch.nn as nn


class LayerNormLSTM(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        batch_first=True,
        ln_type: Optional[str] = "after",
    ):
        super().__init__()

        self.input_dim = input_size
        self.hidden_dim = hidden_size
        self.batch_first = batch_first

        if ln_type == "before":
            # print("Add LayerNorm BEFORE each Linear layer")
            self.linear_w = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.LayerNorm(self.input_dim),
                        nn.Linear(self.input_dim, self.hidden_dim),
                    )
                    for _ in range(4)
                ]
            )
            self.linear_u = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.LayerNorm(self.hidden_dim),
                        nn.Linear(self.hidden_dim, self.hidden_dim),
                    )
                    for _ in range(4)
                ]
            )
        elif ln_type == "after":
            # print("Add LayerNorm AFTER each Linear layer")
            self.linear_w = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(self.input_dim, self.hidden_dim),
                        nn.LayerNorm(self.hidden_dim),
                    )
                    for _ in range(4)
                ]
            )
            self.linear_u = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(self.hidden_dim, self.hidden_dim),
                        nn.LayerNorm(self.hidden_dim),
                    )
                    for _ in range(4)
                ]
            )

        elif ln_type == "none":
            # print("DON'T add LayerNorm")
            self.linear_w = nn.ModuleList(
                [nn.Linear(self.input_dim, self.hidden_dim) for _ in range(4)]
            )
            self.linear_u = nn.ModuleList(
                [nn.Linear(self.hidden_dim, self.hidden_dim) for _ in range(4)]
            )

    def forward(self, inputs, init_states=None):
        if self.batch_first:
            # sequence first (timesteps, batch_size, input_dims)
            inputs = inputs.transpose(0, 1)
        N = inputs.shape[1]

        outputs = []
        if init_states is None:
            h_prev = torch.zeros(N, self.hidden_dim).to(inputs)
            c_prev = torch.zeros(N, self.hidden_dim).to(inputs)
        else:
            h_prev, c_prev = init_states

        outputs = []
        for i, x_t in enumerate(inputs):
            x_f, x_i, x_o, x_c_hat = [linear(x_t) for linear in self.linear_w]
            h_f, h_i, h_o, h_c_hat = [linear(h_prev) for linear in self.linear_u]

            f_t = torch.sigmoid(x_f + h_f)
            i_t = torch.sigmoid(x_i + h_i)
            o_t = torch.sigmoid(x_o + h_o)
            c_t_hat = torch.tanh(x_c_hat + h_c_hat)
            c_prev = torch.mul(f_t, c_prev) + torch.mul(i_t, c_t_hat)
            h_prev = torch.mul(o_t, torch.tanh(c_prev))
            outputs.append(h_prev)

        outputs = torch.stack(outputs, dim=0)
        if self.batch_first:
            # batch first (batch_size, timesteps, input_dims)
            outputs = outputs.transpose(0, 1)

        return outputs, (h_prev, c_prev)


class BaseRnn(nn.Module):
    """LSTM-based network for experiments with Synthetic Normal data and Human Activity."""

    def __init__(
        self,
        input_size: int,
        hidden_dim: int,
        n_layers: int,
        drop_prob: float,
        layer_norm: bool,
        ln_type: str,
        temperature: float = 1.0,
        return_logits: bool = False,
    ) -> None:
        """Initialize model's parameters.

        :param input_size: size of elements in input sequence
        :param output_size: length of the generated sequence
        :param hidden_dim: size of the hidden layer(-s)
        :param n_layers: number of recurrent layers
        :param drop_prob: dropout probability
        """
        super().__init__()

        if layer_norm:
            assert ln_type != "none", "If layer_norm = False, use nn.LSTM"

        self.input_size = input_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        if layer_norm:
            self.lstm = LayerNormLSTM(
                input_size,
                hidden_dim,
                batch_first=True,
                ln_type=ln_type,
            )
        else:
            self.lstm = nn.LSTM(
                input_size, hidden_dim, n_layers, dropout=drop_prob, batch_first=True
            )

        self.linear = nn.Linear(hidden_dim, 1)
        
        # self.head = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, 1),
        # )
        
        self.activation = nn.Sigmoid()
        self.temperature = temperature
        self.return_logits = return_logits

    def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
        """Forward propagation through model.

        :param input_seq: batch of generated sunthetic normal sequences
        :return: probabilities of changes for each sequence
        """
        batch_size = input_seq.size(0)
        lstm_out, _ = self.lstm(input_seq.float())
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        out = self.linear(lstm_out) / self.temperature
        
        # out = self.head(lstm_out) / self.temperature

        if not self.return_logits:
            out = self.activation(out / self.temperature)

        out = out.view(batch_size, -1)
        return out


class CombinedVideoRNN(nn.Module):
    """LSTM-based network for experiments with videos."""

    def __init__(
        self,
        input_dim: int,
        rnn_hidden_dim: int,
        num_layers: int,
        rnn_dropout: float,
        dropout: float,
        layer_norm: bool,
        ln_type: str,
        temperature: float = 1.0,
        return_logits: bool = False,
    ) -> None:
        """Initialize combined LSTM model for video datasets.

        :param input_dim: dimension of the input data (after feature extraction)
        :param rnn_hidden_dim: hidden dimension for LSTM block
        :param rnn_dropuot: dropout probability in LSTM block
        :param dropout: dropout probability in Dropout layer
        """
        super(CombinedVideoRNN, self).__init__()

        if layer_norm:
            assert ln_type != "none", "If layer_norm = False, use nn.LSTM"

        if layer_norm:
            self.rnn = LayerNormLSTM(
                input_size=input_dim,
                hidden_size=rnn_hidden_dim,
                batch_first=True,
                ln_type=ln_type,
            )

        else:
            self.rnn = nn.LSTM(
                input_size=input_dim,
                hidden_size=rnn_hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=rnn_dropout,
            )

        self.fc = nn.Linear(rnn_hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
        # self.relu = nn.ReLU()
        self.activation = nn.Sigmoid()

        self.temperature = temperature
        self.return_logits = return_logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.

        :param x: input torch tensor
        :return: out of the model
        """
        r_out, _ = self.rnn(x)
        out = self.dropout(self.fc(r_out)) / self.temperature

        if not self.return_logits:
            out = self.activation(out)

        return out
