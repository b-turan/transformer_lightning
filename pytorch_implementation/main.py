import datasets
import sh
import torch as th
import transformers
from absl import app, flags
from torch.utils.tensorboard import SummaryWriter

flags.DEFINE_boolean('debug', False, '')
flags.DEFINE_boolean('train', True, " ")
flags.DEFINE_integer('epochs', 5, '')
flags.DEFINE_integer('batch_size', 128, '')
flags.DEFINE_float('lr', 3e-4, '')
flags.DEFINE_float('momentum', .9, '')
flags.DEFINE_string('model', 't5-base', '')
flags.DEFINE_integer('seq_length', 32, '')
flags.DEFINE_integer('percent', 5, '')
flags.DEFINE_integer('num_workers', 8, '')
flags.DEFINE_boolean('is_pretrained', False, '')


FLAGS = flags.FLAGS

# remove and recreate logs folder for development purposes
sh.rm('-r', '-f', 'logs/')
sh.mkdir('logs')
device = th.device('cuda' if th.cuda.is_available() else 'cpu')
print("Running on Device:", device)


def main(path):
    if FLAGS.is_pretrained:
        model = transformers.T5ForConditionalGeneration.from_pretrained(FLAGS.model, torch_dtype="auto").to(device)   
    else:
        tokenizer = transformers.T5Tokenizer.from_pretrained(FLAGS.model)
        start_token_id = tokenizer.convert_tokens_to_ids(['<pad>'])[0] # see transformers/issues/16571
        config = transformers.T5Config(vocab_size=len(tokenizer), decoder_start_token_id=start_token_id)
        model = transformers.T5ForConditionalGeneration(config).to(device)


def convert_for_tokenizer(ds):
    ''' Converts dataset to required format for tokenization '''
    new_ds = {} # re-init dictionary
    new_ds['de'] = [translation['de'] for translation in ds['translation']] # collect src sentences
    new_ds['en'] = [translation['en'] for translation in ds['translation']] # collect trg sentences
    new_ds = datasets.Dataset.from_dict(new_ds)
    return new_ds


def _tokenize(x):
    src_encoding = tokenizer.batch_encode_plus(
            ["translate English to German: " + sentence for sentence in x['en']], 
            max_length=FLAGS.seq_length, 
            padding="longest",
            truncation=True,
            )
    x['src_ids'] = src_encoding.input_ids
    x['attention_mask'] = src_encoding.attention_mask
    x['trg_ids'] = tokenizer.batch_encode_plus(
            x['de'], 
            max_length=FLAGS.seq_length, 
            padding="longest",
            truncation=True,
            )['input_ids']
    return x


def _prepare_ds(tokenizer):
    # available wmt16 language pairs: ['cs-en', 'de-en', 'fi-en', 'ro-en', 'ru-en', 'tr-en']
    ds = datasets.load_dataset('wmt16', 'de-en') # entire dataset {train, validation, test}
    # print("Train Dataset is cut to 2000 samples for development purposes! Remove cutting for full training.")
    train_ds, validation_ds, test_ds = ds['train'][:100000], ds['validation'], ds['test']
    train_ds, validation_ds, test_ds = map(convert_for_tokenizer, (train_ds, validation_ds, test_ds))
    # add tokenized columns to dataset
    train_ds = train_ds.map(_tokenize, batched=True)
    validation_ds = validation_ds.map(_tokenize, batched=True)
    test_ds = test_ds.map(_tokenize, batched=True)
    # convert columns to torch tensors
    train_ds.set_format(type='torch', columns=['src_ids', 'trg_ids', 'attention_mask'])
    validation_ds.set_format(type='torch', columns=['src_ids', 'trg_ids', 'attention_mask'])
    test_ds.set_format(type='torch', columns=['src_ids', 'trg_ids', 'attention_mask'])    
    return train_ds, validation_ds, test_ds 


if __name__ == '__main__':
    app.run(main)
