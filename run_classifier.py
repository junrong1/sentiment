import torch
from torch import nn
import torch.nn.functional as F
import os
import argparse
import logging
import random
import numpy as np
from processor import Semeval_QA_T_Processor, Semeval_QA_EXPT_Processor, Travel_exp_data, Semeval_single_Processor
import tokenization
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.sampler import RandomSampler
from torch.utils.data.distributed import DistributedSampler
from modeling import BertConfig, BertForSequenceClassification
from optimization import BERTAdam
import collections
import visdom

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class InputFeatures(object):

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    label_map = {}
    for i, label in enumerate(label_list):
        label_map[label] = i
    features = []
    tokens_len = []
    for ex_idx, example in enumerate(tqdm(examples)):
        tokens_a = tokenizer.tokenize(example.text_a)
        tokens_b = None
        token_len = len(tokens_a)
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            token_len = token_len + len(tokens_b)
        tokens_len.append(token_len)

        # Truncate the tokens_a and tokens_b to make the total length smaller than max_seq_length
        if tokens_b:
            # Modify the len of tokens_a and tokens_b to smaller than the max_seq_length
            # The length should consider [CLS] [SEP] [SEP]
            # So the max_length - 3
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length-3)
        else:
            # Only Modify the len of tokens_a
            # The length should consider [CLS] [SEP]
            # So the max_length - 2
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:max_seq_length-2]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0

        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.

        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        # tokens [[CLS]]
        # segment_ids [0]
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        # tokens [[CLS], [token_a], [token_a], [SEP]]
        # segment_ids [0, 0, 0, 0]
        tokens.append("[SEP]")
        segment_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)
            # tokens [[CLS], [token_a], [token_a], [SEP], [token_b], [token_b], [SEP]]
            # segment_ids [0, 0, 0, 0, 1, 1, 1]

        # Convert token to id in vocab file
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real token, but 0 for padding
        input_mask = [1]*len(input_ids)

        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_mask) == max_seq_length
        assert len(input_ids) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]

        features.append(InputFeatures(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            label_id=label_id
        ))
    print("max_seq_length: {}, average_seq_lenght: {}".format(max(tokens_len), (sum(tokens_len)/len(tokens_len))))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_len = len(tokens_a) + len(tokens_b)
        if total_len <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def main():
    #rs_writer = SummaryWriter("./log")
    parser = argparse.ArgumentParser()
    viz = visdom.Visdom()

    # Required parameters
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        choices=["semeval_QA_EXPT", "semeval_QA_T", "travel_experience", "semeval_single"],
                        help="Name of the task to train")
    parser.add_argument("--vocab_file", default=None, type=str, required=True,
                        help="The path of BERT pre-trained vocab file")
    parser.add_argument("--init_checkpoint", default=None, type=str, required=True,
                        help="The path of BERT pre-trained .ckpt file")
    parser.add_argument("--bert_config_file", default=None, type=str, required=True,
                        help="The path of BERT .json file")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory of training result")
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The path of training dataset")

    # Other parameters
    parser.add_argument("--train_batch_size", default=32, type=int,
                        help="The size of training batch")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="The size of evaluation batch")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum sentence length of input after WordPiece tonkenization\n"
                             "Greater than the max will be truncated, smaller than the max will be padding")
    parser.add_argument("--local_rank", default=-1, type=int,
                        help="Local_rank for distributed training on gpus")
    parser.add_argument("--seed", default=42, type=int,
                        help="Random seed for initialization")
    parser.add_argument("--accumulate_gradients", default=1, type=int,
                        help="Number of steps to accumulate gradient on (divide the batch_size and accumulate)")
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")
    parser.add_argument('--save_steps', type=int, default=100,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument('--layers', type=int, nargs='+', default=[-2],
                        help="choose the layers that used for downstream tasks, "
                             "-2 means use pooled output, -1 means all layer,"
                             "else means the detail layers. default is -2")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="The number of epochs on training data")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The learning rate of model")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for\n"
                        "0.1 means 10% of training set")
    parser.add_argument('--layer_learning_rate_decay', type=float, default=0.95)
    parser.add_argument('--layer_learning_rate', type=float, nargs='+', default=[2e-5] * 12,
                        help="learning rate in each group")
    parser.add_argument("--do_train", default=False, action="store_true",
                        help="Whether training the data or not")
    parser.add_argument("--do_eval", default=False, action="store_true",
                        help="Whether evaluating the data or not")
    parser.add_argument("--do_predict", default=False, action="store_true",
                        help="Whether predicting the data or not")
    parser.add_argument("--do_lower_case", default=False, action="store_true",
                        help="To lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument("--no_cuda", default=False, action="store_true",
                        help="Whether use the GPU device or not")
    parser.add_argument("--discr", default=False, action='store_true',
                        help="Whether to do discriminative fine-tuning.")
    parser.add_argument('--pooling_type', default=None, type=str,
                        choices=[None, 'mean', 'max'])

    args = parser.parse_args()

    viz = visdom.Visdom()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device %s n_gpu %d distributed training %r", device, n_gpu, bool(args.local_rank != -1))

    if args.accumulate_gradients < 1:
        raise ValueError("Invalid accumulate_gradients parameter: {}, should be >= 1".format(
            args.accumulate_gradients))

    args.train_batch_size = int(args.train_batch_size / args.accumulate_gradients)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    bert_config = BertConfig.from_json_file(args.bert_config_file)

    if args.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length {} because the BERT model was only trained up to sequence length {}".format(
                args.max_seq_length, bert_config.max_position_embeddings))

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)

    # prepare dataloaders
    processors = {
        "semeval_QA_EXPT": Semeval_QA_EXPT_Processor,
        "semeval_QA_T": Semeval_QA_T_Processor,
        "travel_experience": Travel_exp_data,
        "semeval_single": Semeval_single_Processor
    }

    processor = processors[args.task_name]()
    label_list = processor.get_labels()

    tokenizer = tokenization.FullTokenizer(
        vocab_file=args.vocab_file, do_lower_case=args.do_lower_case)

    # training set
    train_examples = processor.get_train_examples(args.data_dir)
    num_train_steps = int(
        len(train_examples) / args.train_batch_size * args.num_train_epochs)

    train_features = convert_examples_to_features(
        train_examples, label_list, args.max_seq_length, tokenizer)
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_examples))
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", num_train_steps)

    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)

    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_data)
    else:
        train_sampler = DistributedSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

    # test set
    if args.do_eval:
        test_examples = processor.get_test_examples(args.data_dir)
        test_features = convert_examples_to_features(
            test_examples, label_list, args.max_seq_length, tokenizer)

        all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in test_features], dtype=torch.long)

        test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        test_dataloader = DataLoader(test_data, batch_size=args.eval_batch_size, shuffle=False)

    # model and optimizer layer version
    """
    model = BertForSequenceClassification(bert_config, len(label_list), args.layers, pooling=args.pooling_type)
    if args.init_checkpoint is not None:
        model.bert.load_state_dict(torch.load(args.init_checkpoint, map_location='cpu'))
    model.to(device)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    no_decay = ['bias', 'gamma', 'beta']
    if args.discr:
        if len(args.layer_learning_rate) > 1:
            groups = [(f'layer.{i}.', args.layer_learning_rate[i]) for i in range(12)]
        else:
            lr = args.layer_learning_rate[0]
            groups = [(f'layer.{i}.', lr * pow(args.layer_learning_rate_decay, 11 - i)) for i in range(12)]
        group_all = [f'layer.{i}.' for i in range(12)]
        no_decay_optimizer_parameters = []
        decay_optimizer_parameters = []
        for g, l in groups:
            no_decay_optimizer_parameters.append(
                {
                    'params': [p for n, p in model.named_parameters() if
                               not any(nd in n for nd in no_decay) and any(nd in n for nd in [g])],
                    'weight_decay_rate': 0.01, 'lr': l
                }
            )
            decay_optimizer_parameters.append(
                {
                    'params': [p for n, p in model.named_parameters() if
                               any(nd in n for nd in no_decay) and any(nd in n for nd in [g])],
                    'weight_decay_rate': 0.0, 'lr': l
                }
            )

        group_all_parameters = [
            {'params': [p for n, p in model.named_parameters() if
                        not any(nd in n for nd in no_decay) and not any(nd in n for nd in group_all)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in model.named_parameters() if
                        any(nd in n for nd in no_decay) and not any(nd in n for nd in group_all)],
             'weight_decay_rate': 0.0},
        ]
        optimizer_parameters = no_decay_optimizer_parameters + decay_optimizer_parameters + group_all_parameters

    else:
        optimizer_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]

    optimizer = BERTAdam(optimizer_parameters,
                         lr=args.learning_rate,
                         warmup=args.warmup_proportion,
                         t_total=num_train_steps)
    """

    model = BertForSequenceClassification(bert_config, len(label_list))
    if args.init_checkpoint is not None:
        model.bert.load_state_dict(torch.load(args.init_checkpoint, map_location='cpu'))
    model.to(device)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    no_decay = ['bias', 'gamma', 'beta']
    optimizer_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]

    optimizer = BERTAdam(optimizer_parameters,
                         lr=args.learning_rate,
                         warmup=args.warmup_proportion,
                         t_total=num_train_steps)

    # train
    output_log_file = os.path.join(args.output_dir, "log.txt")
    print("output_log_file=", output_log_file)
    with open(output_log_file, "w") as writer:
        if args.do_eval:
            writer.write("epoch\tglobal_step\tloss\ttest_loss\ttest_accuracy\n")
        else:
            writer.write("epoch\tglobal_step\tloss\n")

    global_step = 0
    epoch = 0
    best_epoch, best_accuracy = 0, 0
    for _ in trange(int(args.num_train_epochs), desc="Epoch"):
        epoch += 1
        model.train()
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            loss, _ = model(input_ids, segment_ids, input_mask, label_ids)
            print(loss.item())
            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            tr_loss += loss.item()
            viz.line([loss.item()], [global_step], win='tr_loss', update='append')
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()  # We have accumulated enought gradients
                model.zero_grad()
                global_step += 1

                # Save the checkpoint model after each N steps
                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    save_output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(save_output_dir):
                        os.makedirs(save_output_dir)
                    torch.save(model.state_dict(), os.path.join(save_output_dir, "training_args.bin"))
                viz.line([optimizer.get_lr()[0]], [global_step-1], win="lr", update="append")

        # eval_test
        if args.do_eval:
            model.eval()
            test_loss, test_accuracy = 0, 0
            nb_test_steps, nb_test_examples = 0, 0
            with open(os.path.join(args.output_dir, "test_ep_" + str(epoch) + ".txt"), "w") as f_test:
                for input_ids, input_mask, segment_ids, label_ids in tqdm(test_dataloader, desc="Testing"):
                    input_ids = input_ids.to(device)
                    input_mask = input_mask.to(device)
                    segment_ids = segment_ids.to(device)
                    label_ids = label_ids.to(device)

                    with torch.no_grad():
                        tmp_test_loss, logits = model(input_ids, segment_ids, input_mask, label_ids)

                    logits = F.softmax(logits, dim=-1)
                    logits = logits.detach().cpu().numpy()
                    label_ids = label_ids.to('cpu').numpy()
                    outputs = np.argmax(logits, axis=1)
                    for output_i in range(len(outputs)):
                        f_test.write(str(outputs[output_i]))
                        for ou in logits[output_i]:
                            f_test.write(" " + str(ou))
                        f_test.write("\n")
                    tmp_test_accuracy = np.sum(outputs == label_ids)
                    viz.line([tmp_test_loss.item()], [nb_test_steps], win='eval_loss', update='append')
                    test_loss += tmp_test_loss.mean().item()
                    test_accuracy += tmp_test_accuracy

                    nb_test_examples += input_ids.size(0)
                    nb_test_steps += 1

            test_loss = test_loss / nb_test_steps
            test_accuracy = test_accuracy / nb_test_examples
            viz.line([test_accuracy], [nb_test_steps-1], win='test_acc', update='append')
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                best_epoch = epoch
                torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.bin"))

        result = collections.OrderedDict()
        if args.do_eval:
            result = {'epoch': epoch,
                      'global_step': global_step,
                      'loss': tr_loss / nb_tr_steps,
                      'test_loss': test_loss,
                      'test_accuracy': test_accuracy}
        else:
            result = {'epoch': epoch,
                      'global_step': global_step,
                      'loss': tr_loss / nb_tr_steps}

        logger.info("***** Eval results *****")
        with open(output_log_file, "a+") as writer:
            for key in result.keys():
                logger.info("  %s = %s\n", key, str(result[key]))
                writer.write("%s\t" % (str(result[key])))
            writer.write("\n")
        print("The best Epoch is: ", best_epoch)
        print("The best test_accuracy is: ", best_accuracy)


if __name__ == "__main__":
    main()
