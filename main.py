import datasets
import pytorch_lightning as pl
import sh
import tensorboard
import torch as th
import transformers
from absl import app, flags

flags.DEFINE_boolean('debug', False, '')
flags.DEFINE_boolean('train', True, '')
flags.DEFINE_boolean('is_pretrained', False, '')
flags.DEFINE_integer('epochs', 10, '')
flags.DEFINE_integer('training_samples', 5000, 'Number of Training Samples')
flags.DEFINE_integer('batch_size', 64, '')
flags.DEFINE_float('lr', 3e-4, '') # 3e-4 recommended by huggingface docs
flags.DEFINE_float('momentum', .9, '')
flags.DEFINE_string('model', 't5-small', '')
flags.DEFINE_integer('seq_length', 32, '')
flags.DEFINE_integer('percent', 5, '')
flags.DEFINE_integer('num_workers', 8, '')


FLAGS = flags.FLAGS

# remove and recreate logs folder for development purposes
sh.rm('-r', '-f', 'logs')
sh.mkdir('logs')


class TranslationTransformer(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.tokenizer = transformers.T5Tokenizer.from_pretrained(FLAGS.model)
        if FLAGS.is_pretrained:
            self.model = transformers.T5ForConditionalGeneration.from_pretrained(FLAGS.model, torch_dtype="auto")       
        else:
            config = transformers.AutoConfig.from_pretrained(FLAGS.model) # see transformers/issues/14674
            self.model = transformers.T5ForConditionalGeneration(config)
        self.loss = th.nn.CrossEntropyLoss(reduction='none')

    def prepare_data(self):
        def _tokenize(x):
            prefix = "translate English to German: "
            src_encoding = self.tokenizer.batch_encode_plus(
                    [prefix + sentence for sentence in x['en']], 
                    max_length=FLAGS.seq_length, 
                    padding="longest",
                    truncation=True,
                    )
            x['src_ids'] = src_encoding.input_ids
            x['attention_mask'] = src_encoding.attention_mask
            x['trg_ids'] = self.tokenizer.batch_encode_plus(
                    x['de'], 
                    max_length=FLAGS.seq_length, 
                    padding="longest",
                    truncation=True,
                    )['input_ids']
            return x

        def convert_for_tokenizer(ds):
            ''' Converts dataset to required format for tokenization '''
            new_ds = {} # re-init dictionary
            new_ds['de'] = [translation['de'] for translation in ds['translation']] # collect src sentences
            new_ds['en'] = [translation['en'] for translation in ds['translation']] # collect trg sentences
            new_ds = datasets.Dataset.from_dict(new_ds)
            return new_ds

        def _prepare_ds():
            # available wmt16 language pairs: ['cs-en', 'de-en', 'fi-en', 'ro-en', 'ru-en', 'tr-en']
            ds = datasets.load_dataset('wmt16', 'de-en') # {train, validation, test}
            print(f"Train Dataset is cut to {FLAGS.training_samples} samples for development purposes! Remove cutting for full training.")
            train_ds, validation_ds, test_ds = ds['train'][:FLAGS.training_samples], ds['validation'], ds['test']
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

        self.train_ds, self.validation_ds, self.test_ds = _prepare_ds()

    def forward(self, src_ids, attention_mask, trg_ids):
        logits = self.model(input_ids=src_ids, attention_mask=attention_mask, labels=trg_ids)
        return logits

    def training_step(self, batch, batch_idx):
        src_ids, attention_mask, trg_ids = batch['src_ids'], batch['attention_mask'], batch['trg_ids']
        loss = self.model(input_ids=src_ids, attention_mask=attention_mask, labels=trg_ids).loss
        return {'loss': loss, 'log': {'train_loss': loss}}

    def validation_step(self, batch, batch_idx):
        src_ids, attention_mask, trg_ids = batch['src_ids'], batch['attention_mask'], batch['trg_ids']
        loss = self.model(input_ids=src_ids, attention_mask=attention_mask, labels=trg_ids).loss

        # calc SacreBLEU metric
        # TODO: unefficient to calc loss and output_seq seperately
        # TODO: add check_test_every_n_epoch to trainer to avoid bleu metric in every epoch
        # see issue for model.generate: 
        # https://github.com/huggingface/transformers/issues/12503
        pred_seq = self.model.generate(src_ids, 
                                        attention_mask=attention_mask
                                        # do_sample=True, 
                                        # top_p=0.84, 
                                        # top_k=100, 
                                        # max_length=FLAGS.seq_length
                                        ) # encoded translation of src sentences
        trg_decoded = self.tokenizer.batch_decode(trg_ids, skip_special_tokens=True) # decoded trg sentences 
        pred_seq_decoded = self.tokenizer.batch_decode(pred_seq, skip_special_tokens=True) # decoded output translation 

        # original paper of SacreBLEU by Matt Post (2018):
        # https://arxiv.org/pdf/1804.08771.pdf
        # additional material: 
        # hugging face on SacreBLEU score https://www.youtube.com/watch?v=M05L1DhFqcw
        sacre_bleu = datasets.load_metric('sacrebleu') # 'bleu' also possible
        pred_list = [[sentence] for sentence in pred_seq_decoded]
        trg_list = [[sentence] for sentence in trg_decoded]
        sacre_bleu_score = sacre_bleu.compute(predictions=pred_list, references=trg_list)
        return {'loss': loss, 'sacre_bleu_score': sacre_bleu_score}

    def validation_epoch_end(self, outputs):
        loss = th.tensor([o['loss'] for o in outputs]).mean()
        sacre_bleu = th.tensor([o['sacre_bleu_score']['score'] for o in outputs]).mean()
        self.log("val_loss", loss)
        self.log('bleu_score', sacre_bleu)

    def train_dataloader(self):
        return th.utils.data.DataLoader(
                self.train_ds,
                batch_size=FLAGS.batch_size,
                drop_last=True,
                shuffle=True,
                num_workers=FLAGS.num_workers
                )

    def val_dataloader(self):
        return th.utils.data.DataLoader(
                self.validation_ds,
                batch_size=FLAGS.batch_size,
                drop_last=False,
                shuffle=True,
                num_workers=FLAGS.num_workers
                )

    def configure_optimizers(self):
        return th.optim.Adam(
            self.parameters(),
            lr=FLAGS.lr,
            )


def main(_):
    model = TranslationTransformer()
    model_size = sum(t.numel() for t in model.parameters())
    print(f"Model size: {model_size/1000**2:.1f}M parameters")

    trainer = pl.Trainer(
        default_root_dir='logs',
        gpus=(1 if th.cuda.is_available() else 0),
        accelerator='gpu',
        max_epochs=FLAGS.epochs,
        fast_dev_run=FLAGS.debug,
        logger=pl.loggers.TensorBoardLogger('logs/', name='t5_en_de', version=0),
    )

    if FLAGS.train:
        trainer.fit(model)
    else:
        trainer.validate(model, verbose=True)
    

if __name__ == '__main__':
    app.run(main)
