# CS6910 Deep Learning Assignment 3

## Required Libraries
1. `torch`
2. `random`
4. `numpy`
5. `matplotlib`
6. `pandas`

# Classes in Sequence Learning
## `Encoder`
    Init Parameters:
    ----------------
    input_size : english_char_count
    embedding_size : size of each embedding vector
    hidden_size : size of hidden state vector
    num_layers : number of recurrent layers of RNN
    p : dropout probability
    bi_directional : flag to set bidirection in the RNN cell used
    cell_type : type of RNN to be used in the encoder

    Input:

    x: A tensor of shape (seq_length, N), where seq_length is the length of the longest string in the batch, and N is the batch size.
    Output:

            A tensor of shape (seq_len, N, hidden_size * D), where seq_len represents the sequence length, N is the batch size, hidden_size is the size of the hidden state, and D is a scaling factor that depends on whether bidirectional processing is enabled (D = 2 if bidirectional, otherwise D = 1).
            hidden: A tensor of shape (num_layers * D, N, hidden_size), where num_layers is the number of RNN layers stacked, D is the scaling factor, and hidden_size is the size of the hidden state.
            If the RNN class is specified as "LSTM," the following information can be added:

            cell: A tensor of shape (num_layers * D, N, hidden_size), which represents the cell state if the RNN class is LSTM. The cell state is only present in LSTM cells and is used to carry long-term information throughout the sequence. Where the input tensor x is fed through the RNN layers, and the outputs, hidden, and cell tensors are the corresponding outputs of the RNN after processing the input sequence.
## `Decoder`
    Init Parameters:
    ---------------
    input_size: target_char_count
    embedding_size: size of each embedding vector
    hidden_size: size of hidden state vector
    output_size: number of output features in fully connected layer, here target_char_count
    num_layers : number of recurrent layers of RNN
    p : dropout probability
    bidirectional : flag to set bidirection in the RNN cell used
    cell_type: type of RNN to be used in the encoder

    Input:

    x: A tensor of shape (N), representing the input sequence for the RNN.
    hidden: A tensor of shape (num_layers * D, N, hidden_size), where num_layers is the number of RNN layers stacked, D is the scaling factor (equal to 2 if bidirectional, otherwise 1), and hidden_size is the size of the hidden state.
    cell: A tensor of shape (num_layers * D, N, hidden_size), representing the cell state. This tensor is present if the RNN class is LSTM; otherwise, it can be ignored.
    Outputs:

    predictions: A tensor of shape (N, target_char_count), containing the predicted values for each element in the batch. target_char_count represents the number of target characters or classes that the RNN predicts.
    hidden: A tensor of shape (num_layers * D, N, hidden_size), representing the updated hidden state after processing the input sequence.
    cell: A tensor of shape (num_layers * D, N, hidden_size), which is only present if the RNN class is specified as "LSTM." This tensor represents the updated cell state after processing the input sequence in an LSTM cell.
## `AttentionDecoder`
    Decoder specially when attention is implemented
    Init Parameters:
    ---------------
    embedding_size: size of each embedding vector
    hidden_size: size of hidden state vector
    output_size: number of output features in fully connected layer, here target_char_count
    num_layers : number of recurrent layers of RNN
    p : dropout probability
    max_length : maximum length of the input word (from encoder) for which this decoder is able to handle - by using attention mechanism
    bidirectional : flag to set bidirection in the RNN cell used
    cell_type: type of RNN to be used in the encoder

    Input:

    input: A tensor of shape (N), representing the input sequence for the RNN.
    hidden: A tensor of shape (num_layers * D, N, hidden_size), where num_layers is the number of RNN layers stacked, D is the scaling factor (equal to 2 if bidirectional, otherwise 1), and hidden_size is the size of the hidden state.
    cell: A tensor of shape (num_layers * D, N, hidden_size), representing the cell state. This tensor is present if the RNN class is LSTM; otherwise, it can be ignored.
    encoder_outputs: A tensor of shape (seq_len, N, hidden_size * D), containing the output representations from the encoder. seq_len is the length of the encoder outputs, N is the batch size, hidden_size is the size of the hidden state, and D is the scaling factor (equal to 2 if bidirectional, otherwise 1).
    Outputs:

    prob: A tensor of shape (N, target_char_count), representing the predicted probabilities for each element in the batch. target_char_count represents the number of target characters or classes that the RNN predicts.
    hidden: A tensor of shape (num_layers * D, N, hidden_size), representing the updated hidden state after processing the input sequence.
    attn_weights: A tensor of shape (1, N, max_len), representing the attention weights. max_len is the maximum length of the encoder outputs.
    If the RNN class is specified as "LSTM," the following information can be added:

    cell: A tensor of shape (num_layers * D, N, hidden_size), representing the updated cell state after processing the input sequence in an LSTM cell.
## `Seq2Seq`
    This Class combines the Encoder and Decoder Classes seen above. 
    Init Parameters:
    ---------------
    encoder: Encoder class object
    decoder: Decoder class object

    Input:
    -----
    source : torch.Tensor of shape (source seq_len, N) where source seq_len = len(longest word in the batch) if attention is not used, 
                                                                                                        else seq_len = max_length 
    target : torch.Tensor of shape (target seq_len, N)
    teacher_forcing : A boolean value to indicate if the teacher forcing should be used.
    
    Output:
    ------
    outputs : torch.Tensor of shape(target seq_len, batch_size, target_char_count)

# Training
## Steps involved in training
1. Initialize `Encoder`, `Decoder`, `AttnDecoder` and `Seq2Seq` objects
2. Load the data using from `Aksharantar data set`
3. Call `neural_network()` method

Configuration for the best model using attention:   
`--batch_size 64 --epochs 15 --learning_rate 0.001 --embedding_size 64 --encoder_layers 3 --decoder_layers 3 --enc_dropout 0.2 --dec_dropout 0.2 --hidden_size 512 --rnn_class LSTM --bi_directional --attention`
