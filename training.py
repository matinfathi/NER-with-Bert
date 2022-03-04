from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForTokenClassification, logging
from keras.preprocessing.sequence import pad_sequences
from seqeval.metrics import accuracy_score
from tqdm import tqdm, trange
import torch.nn.functional as F
import pandas as pd
import torch

logging.set_verbosity_error()


def read_data(path):
    """
    Read data from dataset file.
    """
    # Open file and split each sentence
    with open(path, 'r') as f:
        data = f.read().split('\n\n')[:-1]

    # Create an empty dataframe to append each sentence to this dataframe
    df = pd.DataFrame([], columns=['sent',
                                   'word',
                                   'tag'])

    # Iterate over data and read each sentence
    for i, line in tqdm(enumerate(data), total=len(data)):
        temp_sent, temp_lab = [], []

        line_split = line.split()  # split words and tags

        for idx, item in enumerate(line_split):  # iterate over words
            # if the index of the word is even its a tag and
            # if the index is odd its a word
            if idx % 2:
                temp_sent.append(item)
            else:
                temp_lab.append(item)

        # Append the extracted sentence and tags to the dataframe
        df_temp = pd.DataFrame({'sent': [f'sentence {i + 1}'] * len(temp_sent),
                                'word': temp_sent,
                                'tag': temp_lab, })

        df = df.append(df_temp,
                       ignore_index=True, )

    return df


def create_tag_dictionary(df):
    """
    Create a dictionary of unique tags and idx of them.
    """
    unique_tags = df['tag'].unique().tolist()  # extract unique tags
    tag2idx = dict(zip(unique_tags, range(len(unique_tags))))  # create dictionary
    tag2idx['[CLS]'] = len(unique_tags)  # add [CLS] to dictionary
    tag2idx['[SEP]'] = len(unique_tags) + 1  # add [SEP] to dictionary
    tag2idx['[PAD]'] = len(unique_tags) + 2  # add [PAD] to dictionary
    idx2tag = {value: key for key, value in tag2idx.items()}  # create a reverese dictionary for validation

    return tag2idx, idx2tag


def dataframe2list(df, tag2idx):
    """
    Convert our dataset to a list of lists and replace string tags with ids.
    """
    # Group each sentence in dataframe and add [CLS] and [SEP] to the sentence
    sentences = df.groupby('sent')['word'].apply(list)
    sentences = sentences.tolist()
    for sent in sentences:
        sent.insert(0, '[CLS]')
        sent.insert(len(sent), '[SEP]')

    # Replace string tags with ids and group tags related to each sentence
    df['tag'] = df['tag'].replace(tag2idx)
    tags = df.groupby('sent')['tag'].apply(list)
    tags = tags.tolist()
    for tag in tags:
        tag.insert(0, tag2idx['[CLS]'])
        tag.insert(len(tag), tag2idx['[SEP]'])

    return sentences, tags


def create_input_embeddigs(sentences, tags, tag2idx):
    """
    Create input ids and attention mask for tokenclassification task.
    """

    # Extract the maximum length of sentences for padding
    max_len = max([len(item) for item in tr_sentences])

    # Pad sentences and convert words to input ids
    input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(sent) for sent in sentences],
                              maxlen=max_len,
                              dtype="long",
                              truncating="post",
                              padding="post", )
    # Pad labels as well with [PAD] tag
    labels = pad_sequences(tags,
                           maxlen=max_len,
                           dtype="long",
                           truncating="post",
                           padding="post",
                           value=tag2idx['[PAD]'], )
    # Create attention mask for sentences, its 0 where the words is [PAD] and
    # its 1 when its not [PAD]
    attention_masks = [[int(i > 0) for i in ii] for ii in input_ids]

    return input_ids, attention_masks, labels


def create_dataloader(input_ids, attention_masks, labels):
    """
    Create a torch dataloader with dataset.
    """
    # Convert list of input ids and attention masks and labels to tensor
    inputs = torch.tensor(input_ids)
    tags = torch.tensor(labels)
    masks = torch.tensor(attention_masks)

    # Create a torch dataset with two input and a label
    dataset = TensorDataset(inputs, masks, tags)
    # Define a random sampler for roch dataloader
    sampler = RandomSampler(dataset)
    # Create a torch dataloader with our input and label
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=32, drop_last=True)

    return dataloader


def train(train_dataloader, bftc, epochs=6):
    """
    Trainig process(forward and backward propagation) and print the loss.
    """

    # Iterate over training dataset
    for _ in trange(epochs, desc="Epoch"):
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch

            # forward pass
            outputs = bftc(b_input_ids,
                           token_type_ids=None,
                           attention_mask=b_input_mask,
                           labels=b_labels)
            loss, scores = outputs[:2]

            # backward pass
            loss.backward()

            # track train loss
            tr_loss += loss.item()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1

            # gradient clipping
            torch.nn.utils.clip_grad_norm_(parameters=bftc.parameters(), max_norm=1.0)

            # update parameters
            optimizer.step()
            optimizer.zero_grad()

        # print train loss per epoch
        print(" ---> Train loss: {}".format(tr_loss / nb_tr_steps))

    return bftc


def validation(test_dataloader, idx2tag):
    """
    The validating process and print the accuracy
    """
    test_accuracy, test_loss = 0, 0
    y_true = []
    y_pred = []

    # Iterate over test dataset
    for step, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):

        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, label_ids = batch

        with torch.no_grad():
            outputs = bftc(input_ids,
                           token_type_ids=None,
                           attention_mask=input_mask,
                           )
            # For eval mode, the first result of outputs is logits
            logits = outputs[0]

            # Get NER predict result
        logits = torch.argmax(F.log_softmax(logits, dim=2), dim=2)
        logits = logits.detach().cpu().numpy()

        # Get NER true result
        label_ids = label_ids.to('cpu').numpy()

        # Only predict the real word, mark=0, will not calculate
        input_mask = input_mask.to('cpu').numpy()

        # Compare the valuable predict result
        for i, mask in enumerate(input_mask):
            # Real one
            temp_1 = []
            # Predict one
            temp_2 = []

            for j, m in enumerate(mask):
                # Mark=0, meaning its a pad word, dont compare
                if m:
                    if idx2tag[label_ids[i][j]] != "[CLS]" and idx2tag[label_ids[i][j]] != "[SEP]":
                        temp_1.append(idx2tag[label_ids[i][j]])
                        temp_2.append(idx2tag[logits[i][j]])
                else:
                    break

            y_true.append(temp_1)
            y_pred.append(temp_2)

    print("\n---> The accuracy score is: %f" % (accuracy_score(y_true, y_pred)))

    return y_true, y_pred


if __name__ == '__main__':

    # Specify usage of GPU or CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Define variables for training
    train_path = './Dataset/engtrain.bio'
    test_path = './Dataset/engtest.bio'
    path_for_saving_model = './Ner_model1.pth'
    learning_rate = 1e-4
    epochs = 5
    print(f"---> Using {str(device).capitalize()} for calculations.")

    # Load train and test data
    print('\n---> Reading the train data:\n')
    df_train = read_data(train_path)
    print('\n---> Reading the test data:\n')
    df_test = read_data(test_path)

    # Create the dictionary for tag ids
    tag2idx, idx2tag = create_tag_dictionary(df_train)

    # convert data to a list of lists of sentences
    tr_sentences, tr_tags = dataframe2list(df_train, tag2idx)
    te_sentences, te_tags = dataframe2list(df_test, tag2idx)

    # Load the Bert model for token classification and specify the number of
    # output labels, also if we use GPU transfer the model to cuda
    print('\n---> Loading the Bert Model:\n')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bftc = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=len(tag2idx))
    if str(device) == 'cuda':
        bftc.cuda()
    print('\n---> The Bert Model has been loaded.\n')

    # Create input ids, attention masks and output labels
    train_inputs, train_masks, train_labels = create_input_embeddigs(tr_sentences, tr_tags, tag2idx)
    test_inputs, test_masks, test_labels = create_input_embeddigs(te_sentences, te_tags, tag2idx)

    # Create torch dataloader for convenient
    train_dataloader = create_dataloader(train_inputs, train_masks, train_labels)
    test_dataloader = create_dataloader(test_inputs, test_masks, test_labels)

    # We only fine-tune the classifier in Bert model
    param_optimizer = list(bftc.classifier.named_parameters())
    optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)

    # Start fine-tuning
    print('\n---> Start fine-tune process:\n')
    bftc = train(train_dataloader, bftc, epochs)
    print('\n---> The training is finished!\n')

    torch.save(bftc, path_for_saving_model)
    print('\nThe trained model has been saved!\n')

    # Start the validation process
    print('\n---> Calculating the accuracy of model on test data.\n')
    y_true, y_pred = validation(test_dataloader, idx2tag)