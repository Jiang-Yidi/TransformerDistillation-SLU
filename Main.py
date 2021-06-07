import argparse
import math
import utils
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from model_transformer import *
import numpy as np
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import os
from data import *
from transformer.modeling import BertModel
from torch.nn import MSELoss
from transformers import AutoTokenizer
from optimizer import TransformerOptimizer


IGNORE_ID = -1
def cal_loss(pred, gold, smoothing):
    """Calculate cross entropy loss, apply label smoothing if needed.
    """

    if smoothing > 0.0:
        eps = smoothing
        n_class = pred.size(1)

        # Generate one-hot matrix: N x C.
        # Only label position is 1 and all other positions are 0
        # gold include -1 value (IGNORE_ID) and this will lead to assert error
        gold_for_scatter = gold.ne(IGNORE_ID).long() * gold
        one_hot = torch.zeros_like(pred).scatter(1, gold_for_scatter.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / n_class
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(IGNORE_ID)
        n_word = non_pad_mask.sum().item()
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum() / n_word
    else:
        loss = F.cross_entropy(pred, gold,
                               ignore_index=IGNORE_ID,
                               reduction='elementwise_mean')

    return loss


def init_model(config, embedding_size, batchsize, hidden_size, num_features, num_intents, nhead, nlayers, d_k,
               d_v, d_model, d_inner):
    num_features = 256
    model = TransformerModelSpeech(config, num_intents, embedding_size, batchsize, num_features, nhead, hidden_size,
                                   nlayers, d_k, d_v, d_model, d_inner).to(device)

    return model


def save_models(args, model_name, model):
    torch.save(model.state_dict(), os.path.join(args.model_dir, 'Transformer-{}'.format(model_name)))


def load_models(args, model_name, model, tst_mode=True):
    model.load_state_dict(torch.load(os.path.join(args.model_dir, 'Transformer-{}'.format(model_name))))
    if tst_mode:
        model.eval()
    return model

def init_logger(log_file=None):
    log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.NOTSET)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]

    if log_file and log_file != '':
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
    return logger



def train(args):
    embedding_size = args.embedding_size
    hidden_size = args.hidden_size
    torch.manual_seed(111)
    torch.cuda.manual_seed(111)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ## Generated variables
    args.model_dir = "models/model-{}-data_type_{}-lr_{}/".format(args.data, args.datatype, args.learning_rate)
    logdir = "logs/data-{}-data_type_{}-emb_size_{}-hid_size_{}-lr_{}/".format(args.data, args.datatype, embedding_size,
                                                                               hidden_size, args.learning_rate)
    logfile = "{}/output.logs".format(logdir)
    writer = SummaryWriter(logdir="{}/".format(logdir))
    logger = init_logger(log_file=logfile)
    logger.info("Generated logger information")

    ## Training
    logger.info("Training starts")
    logger.info("splitting data in train and validation-set")
    logger.info("Model initialization for training and setting up the training environment")
    num_intents = 31
    args.num_features = 768
    config_path = './no_unfreezing.cfg'
    config = read_config(config_path)
    train_dataset, valid_dataset, test_dataset = get_SLU_datasets(config)

    model = init_model(config, embedding_size, args.batch_size, hidden_size, args.num_features,
                       num_intents, args.nhead, args.nlayers, args.d_k, args.d_v, args.d_model, args.d_inner)

    optimizier = TransformerOptimizer(
        torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-09),
        args.k,
        args.d_model,
        args.warmup_steps)

    loss_mse = MSELoss()

    teacher_model = BertModel.from_pretrained(args.teacher_model)
    teacher_model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)

    best_epoch = 0
    max_acc = 10

    if not os.path.exists(args.model_dir):  # save initial model
        os.makedirs(args.model_dir)
        save_models(args, "inital-{}.pt".format(best_epoch), model)


    for epoch in range(args.num_train_epochs):
        model.train()  # Turn on the train model
        train_total_loss = 0
        train_intent_loss = 0
        train_attn_loss = 0
        train_hidden_loss = 0
        train_intent_acc = 0
        num_examples = 0
        logger.info("Run training for epoch # {}".format(epoch))
        for idx, batch in enumerate(train_dataset.loader):

            x_data, y_data, text = batch  # x2->'input_id'; m2->'attention_mask'
            batch_size = len(x_data)

            output2, intent_loss, intent_acc, student_atts, student_reps, length, score = model(x_data, y_data)

            att_loss = 0.
            rep_loss = 0.

            tokens = tokenizer.batch_encode_plus(text, max_length=length, padding='max_length', return_tensors='pt',
                                                 truncation=True)
            x2 = tokens['input_ids']
            t2 = tokens['token_type_ids']
            m2 = tokens['attention_mask']

            x2 = x2.to(device)
            t2 = t2.to(device)
            m2 = m2.to(device)
            teacher_reps, teacher_atts, _ = teacher_model(x2, t2, m2)  # input_ids, token_type_ids=None, attention_mask=input_mask

            # BERT Base has 12 layers and 12 heads, resulting in a total of 12 x 12 = 144 distinct attention mechanisms.
            teacher_reps = [teacher_rep.detach() for teacher_rep in teacher_reps]  # speedup 1.5x
            teacher_atts = [teacher_att.detach() for teacher_att in teacher_atts]
            teacher_layer_num = len(teacher_atts)  # 12
            student_layer_num = len(student_atts)  # 4
            assert teacher_layer_num % student_layer_num == 0
            layers_per_block = int(teacher_layer_num / student_layer_num)  # 3
            new_teacher_atts = [teacher_atts[i * layers_per_block + layers_per_block - 1]
                                for i in range(student_layer_num)]

            for student_att, teacher_att in zip(student_atts, new_teacher_atts):
                student_att = torch.where(student_att <= -1e2, torch.zeros_like(student_att).to(device),
                                          student_att)
                teacher_att = torch.where(teacher_att <= -1e2, torch.zeros_like(teacher_att).to(device),
                                          teacher_att)  # [64,12,25,25][batch_size, num_heads, seq_length, seq_length]
                student_att = student_att.reshape(batch_size, args.nhead, length, length)
                teacher_att = teacher_att.to(device)
                student_att = torch.mean(student_att, 1, False)
                teacher_att = torch.mean(teacher_att, 1, False)
                att_loss += loss_mse(student_att, teacher_att)

            new_teacher_reps = [teacher_reps[i * layers_per_block] for i in range(student_layer_num + 1)]
            new_student_reps = student_reps

            for student_rep, teacher_rep in zip(new_student_reps, new_teacher_reps):
                rep_loss += loss_mse(student_rep, teacher_rep)

            #total loss
            loss = args.alpha1*att_loss + args.alpha2* rep_loss + args.alpha3* intent_loss

            optimizier.zero_grad()
            loss.backward()
            optimizier.step()

            num_examples += batch_size
            batch_size = len(x_data)
            num_examples += batch_size
            train_intent_loss += intent_loss.cpu().data.numpy().item() * batch_size
            train_intent_acc += intent_acc.cpu().data.numpy().item() * batch_size

            train_total_loss += loss.cpu().data.numpy().item() * batch_size
            train_intent_loss += intent_loss.cpu().data.numpy().item() * batch_size
            train_attn_loss += att_loss.cpu().data.numpy().item() * batch_size
            train_hidden_loss += rep_loss.cpu().data.numpy().item() * batch_size

            if idx % args.logging_steps == 0:  # only print for some steps
                logger.info(
                    "Epoch: {0:2d} \t Batch id:{1:3d} \t loss: {2:2.4f} \t Accu:{3:2.3f}".format(epoch, idx, loss,
                                                                                                 intent_acc * 100))

        train_intent_acc /= num_examples
        print("==============train intent acc:============== " + str(train_intent_acc))
        train_total_loss /= num_examples
        train_intent_loss /= num_examples
        train_attn_loss /= num_examples
        train_hidden_loss /= num_examples
        logger.info(
            "train total loss: {0:2.4f} \t train_intent_loss: {1:2.4f} \t train_attn_loss: {2:2.4f} \t train_hidden_loss: {3:2.4f}".format(
                train_total_loss, train_intent_loss, train_attn_loss, train_hidden_loss))

        ##Testing Done Here
        num_examples = 0
        model.eval()
        test_intent_acc = 0
        for idx, batch in enumerate(test_dataset.loader):
            loss = 0
            x_data, y_data, _ = batch
            batch_size = len(x_data)
            num_examples += batch_size
            output2_test, intent_loss_test, intent_acc_test, atts, reps, L, score_tst = model(x_data, y_data)
            test_intent_acc += intent_acc_test.cpu().data.numpy().item() * batch_size

        test_intent_acc /= num_examples
        # print("==============test intent acc:============== " + str(test_intent_acc*100))
        logger.info("Epoch: {0:3d}\t test intent acc:{1:2.3f}".format(epoch, test_intent_acc))
        if test_intent_acc * 100 > max_acc:
            max_acc = test_intent_acc * 100
            best_epoch = epoch
            save_models(args, "transformer-{}.pt".format(best_epoch), model)
        logger.info("Best Epoch: {0:3d}\t Best Accu:{1:2.3f}".format(best_epoch, max_acc))
        logger.info("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--datatype", default="speech", type=str,
                        help="The name of datatype task to train")  # speech, text, speechtext
    parser.add_argument("--data", default="FluentAI", type=str,
                        help="data type to save, load model")  # FluentAI, ATIS, DSTC
    parser.add_argument("--teacher_model", default="./bert-base-uncased", type=str)

    parser.add_argument('--seed', type=int, default=111, help="random seed for initialization")
    parser.add_argument("--batch_size", default=, type=int, help="Batch size for training and evaluation.")
    parser.add_argument("--max_seq_len", default=50, type=int,
                        help="The maximum total input sequence length after tokenization.")
    parser.add_argument('--num_features', type=int, default=48, help="number of MFCC features")

    parser.add_argument('--max_length_text', type=int, default=20, help="max_length_text")

    parser.add_argument("--learning_rate", default=, type=float,
                        help="The initial learning rate for Adam.")  # lr was 5 originally
    parser.add_argument("--num_train_epochs", default=100, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--logging_steps', type=int, default=, help="Log every X updates steps.")

    parser.add_argument('--hidden_size', type=int, default=256, help="Hidden size")
    parser.add_argument('--embedding_size', type=int, default=256, help="Embedding size.")

    parser.add_argument("--model_dir", default="models_ATIS_text", type=str, help="The input prediction dir")

    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the test set.")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--nhead", default=8, type=int, help="nhead.")
    parser.add_argument("--nlayers", default=6, type=int, help="nlayers.")
    parser.add_argument("--d_k", default=64, type=int, help="d_k.")
    parser.add_argument("--dropout", default=0.1, type=float, help="dropout.")
    parser.add_argument("--d_v", default=64, type=int, help="d_v.")
    parser.add_argument("--d_model", default=512, type=int, help="d_model.")
    parser.add_argument("--d_inner", default=768, type=int, help="d_inner.")
    parser.add_argument('--label_smoothing', default=, type=float,
                        help='smoothing')
    ##loss parameter
    parser.add_argument('--alpha1', default=, type=float,
                        help='alpha1')
    parser.add_argument('--alpha2', default=, type=float,
                        help='alpha2')
    parser.add_argument('--alpha3', default=, type=float,
                        help='alpha3')

    # optimizer
    parser.add_argument('--k', default=0.95, type=float,
                        help='tunable scalar multiply to learning rate')
    parser.add_argument('--warmup_steps', default=18000, type=int,
                        help='warmup steps')
    args = parser.parse_args()
    train(args)

