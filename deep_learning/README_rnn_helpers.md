# rnn_helpers.py

This module includes small PyTorch classes implementing different recurrent neural networks. RNNs are well-suited for
processing sequential data such as text, audio or time series.

## Classes

- **`SimpleRNNModel`** – uses the basic `nn.RNN` layer. The final hidden state of the sequence feeds into a linear
  layer to produce the output.
- **`LSTMModel`** – employs an LSTM layer which keeps separate hidden and cell states, allowing the network to learn
  longer-range dependencies.
- **`GRUModel`** – similar to the LSTM but with a simplified gating mechanism. Useful when you want fewer parameters.

## Syntax Points

All models inherit from `nn.Module`. The `forward` methods return the output after processing the last time step. Batch
first mode (`batch_first=True`) expects tensors shaped `(batch, seq, features)`.

## Theory

Recurrent networks maintain internal state across time steps. LSTMs and GRUs introduce gates that control information
flow, mitigating the vanishing gradient problem and enabling the capture of long-term patterns. They are widely used in
language modelling, speech recognition and other sequence prediction tasks.

Classic RNNs struggle with long dependencies because repeated multiplications of
the recurrent weight matrix shrink or explode gradients. LSTMs and GRUs address
this by adding gating mechanisms that regulate how information enters, leaves and
is forgotten from the cell state. These gates allow the network to maintain a
stable memory over many time steps, making them effective for natural language
and time series data where context matters.
