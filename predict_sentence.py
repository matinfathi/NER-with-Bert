# Import libraries
from transformers import BertTokenizer, BertForTokenClassification
from keras.preprocessing.sequence import pad_sequences
import torch.nn.functional as F
import torch
import os

# The target sentence for predicting its tokens.
sentence = 'it was a stupid movie starring tom hanks'

if __name__ == '__main__':

    # Define the dictionary for converting the classifications to tags
    idx2tag = {
        0: 'O',
        1: 'B-ACTOR',
        2: 'I-ACTOR',
        3: 'B-YEAR',
        4: 'B-TITLE',
        5: 'B-GENRE',
        6: 'I-GENRE',
        7: 'B-DIRECTOR',
        8: 'I-DIRECTOR',
        9: 'B-SONG',
        10: 'I-SONG',
        11: 'B-PLOT',
        12: 'I-PLOT',
        13: 'B-REVIEW',
        14: 'B-CHARACTER',
        15: 'I-CHARACTER',
        16: 'B-RATING',
        17: 'B-RATINGS_AVERAGE',
        18: 'I-RATINGS_AVERAGE',
        19: 'I-TITLE',
        20: 'I-RATING',
        21: 'B-TRAILER',
        22: 'I-TRAILER',
        23: 'I-REVIEW',
        24: 'I-YEAR',
        25: '[CLS]',
        26: '[SEP]',
        27: '[PAD]',
    }

    # Specify the device for calculations
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Path to fine-tuned model
    model_path = './Ner_model.pth'

    # if there is a fine-tuned model we use it, else
    # we use bert without fine-tuning
    # and it is obvious that without fine-tuning
    # the model won't have good predictions so it is
    # recommended to use a fine-tuned model.
    if os.path.isfile(model_path):
        bftc = torch.load(model_path)
        if torch.cuda.is_available():
            bftc.cuda()
    else:
        bftc = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=len(idx2tag))

    # add [CLS] and [SEP] to the sentence
    tok_sentence = sentence.split()
    tok_sentence.insert(0, '[CLS]')
    tok_sentence.insert(len(tok_sentence), '[SEP]')

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # pad and tokenize sentence
    input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(tok_sentence)],
                              maxlen=49,
                              dtype="long",
                              truncating="post",
                              padding="post", )
    # Create attention mask
    attention_masks = [[int(i > 0) for i in ii] for ii in input_ids]

    inputs = torch.tensor(input_ids).to(device)
    masks = torch.tensor(attention_masks).to(device)

    # Predict outputs with model
    with torch.no_grad():
        outputs = bftc(inputs,
                       token_type_ids=None,
                       attention_mask=masks,
                       )
        logits = outputs[0]

    logits = torch.argmax(F.log_softmax(logits, dim=2), dim=2)
    logits = logits.detach().cpu().numpy()

    tags = []

    # Convert predicted outputs to tag strings
    for j, m in enumerate(masks[0]):
        if m:
            tags.append(idx2tag[logits[0][j]])
        else:
            break

    print(tags[1:-1])
