import argparse
import logging
import os
from functools import partial
import scipy
import numpy as np
import torch
from torch.utils.data import RandomSampler, SequentialSampler
from tqdm import tqdm
import time
import structs
import utils
from utils import clustering, visualize_goals_2D
from modeling.vectornet import VectorNet

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
tqdm = partial(tqdm, dynamic_ncols=True)


def eval_instance_argoverse(batch_size, args, pred, score, pred_int, score_int, mapping, file2pred, file2score, file2pred_int, file2score_int, city_name, file2labels, DEs, iter_bar,id_with_modes):
    if args.clustering:
        for i,id in enumerate(id_with_modes):
            a_pred = pred[id]
            a_pred_int = pred_int[i] 
            assert a_pred.shape == (args.mode_num, args.future_frame_num, 2)
            file_name = int(os.path.split(mapping[id]['file_name'])[1][:-4])
            file2pred[file_name] = a_pred
            file2score[file_name] = score[id] 
            file2pred_int[file_name] = a_pred_int
            file2score_int[file_name] = score_int[i]
            city_name[file_name] = mapping[id]['city_name']
            if not args.do_test:
                file2labels[file_name] = mapping[id]['origin_labels']
    else:
        for i in range(batch_size):
            a_pred = pred[i]
            assert a_pred.shape == (args.mode_num, args.future_frame_num, 2)
            file_name_int = int(os.path.split(mapping[i]['file_name'])[1][:-4])
            file2pred[file_name_int] = a_pred
            file2score[file_name_int] = score[i] 
            city_name[file_name_int] = mapping[i]['city_name']
            if not args.do_test:
                file2labels[file_name_int] = mapping[i]['origin_labels']

    if not args.do_test:
        DE = np.zeros([batch_size, args.future_frame_num])
        for i in range(batch_size):
            origin_labels = mapping[i]['origin_labels']
            for j in range(args.future_frame_num):
                DE[i][j] = np.sqrt((origin_labels[j][0] - pred[i, 0, j, 0]) ** 2 + (
                        origin_labels[j][1] - pred[i, 0, j, 1]) ** 2)
        DEs.append(DE)
        miss_rate = 0.0
        if 0 in utils.method2FDEs:
            FDEs = utils.method2FDEs[0]
            miss_rate = np.sum(np.array(FDEs) > 2.0) / len(FDEs)

        iter_bar.set_description('Iter (MR=%5.3f)' % (miss_rate))


def do_eval(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    print("Loading Evalute Dataset", args.data_dir)
    if args.argoverse:
        from dataset_argoverse import Dataset
    eval_dataset = Dataset(args, args.eval_batch_size)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.eval_batch_size,
                                                  sampler=eval_sampler,
                                                  collate_fn=utils.batch_list_to_batch_tensors, 
                                                  pin_memory=False)
    model = VectorNet(args)
    print('torch.cuda.device_count', torch.cuda.device_count())

    logger.info("***** Recover model: %s *****", args.model_recover_path)
    if args.model_recover_path is None:
        raise ValueError("model_recover_path not specified.")

    model_recover = torch.load(args.model_recover_path)
    model.load_state_dict(model_recover)

    if 'set_predict-train_recover' in args.other_params and 'complete_traj' in args.other_params:
        model_recover = torch.load(args.other_params['set_predict-train_recover'])
        utils.load_model(model.decoder.complete_traj_cross_attention, model_recover, prefix='decoder.complete_traj_cross_attention.')
        utils.load_model(model.decoder.complete_traj_decoder, model_recover, prefix='decoder.complete_traj_decoder.')

    model.to(device)
    model.eval()
    file2pred = {}
    file2pred_int = {}
    file2score = {}
    file2score_int = {}
    city_name = {}
    pred_intention = [] 
    agent_dir_int_var_list = []
    agent_dir_var_list = []
    file2labels = {}
    opposite_dir_batch = 0 # [8, 16] how many modes going in opposite direction per sequence
    iter_bar = tqdm(eval_dataloader, desc='Iter (loss=X.XXX)')
    DEs = []
    length = len(iter_bar)
    argo_pred = structs.ArgoPred()
    max_guesses = args.mode_num

    for step, batch in enumerate(iter_bar): 
        pred_trajectory, pred_score, _ = model(batch, device)  
        pred_intention = []
        pred_intention_score =  []
        id_with_modes = []
        mapping = batch
        batch_size = pred_trajectory.shape[0]
        for i in range(batch_size): 
            #if mapping[i]['file_name'].split('/')[-1] not in ['32683.csv']: 
            #    continue
            assert pred_trajectory[i].shape == (args.mode_num, args.future_frame_num, 2)
            assert pred_score[i].shape == (args.mode_num,)
            if args.clustering:
                mapping[i]['element_in_batch'] = i 
                if mapping[i]['file_name'].split('/')[-1] in ['1825.csv','20880.csv','13067.csv','7214.csv','34487.csv', '38044.csv']:
                    continue
                pred_intention_ids, cluster_probs, agent_dir_var,agent_dir_int_var, opposite_dir, vis_clusters = clustering(mapping[i], mapping[i]['vis.goals_2D'], 
                                mapping[i]['vis.scores'], args.future_frame_num, mapping[i]['vis.predict_trajs'], max_guesses) 

                if args.visualize:  
                    mapping[i]['element_in_batch'] = i
                    visualize_goals_2D(mapping[i], mapping[i]['vis.goals_2D'], mapping[i]['vis.scores'], args.future_frame_num,  
                                        vis_clusters,
                                        labels=mapping[i]['vis.labels'],
                                        labels_is_valid=mapping[i]['vis.labels_is_valid'],
                                        predict=mapping[i]['vis.predict_trajs']) 
                if len(cluster_probs) == 0:
                    print('No clusters found for ', mapping[i]['file_name'])
                    continue
            
                # Evaluate only those scenarios with more than one intention
                if True: #len(pred_intention_ids) > 1:
                    agent_dir_var_list.append(agent_dir_var)
                    agent_dir_int_var_list.append(agent_dir_int_var)
                    opposite_dir_batch += (opposite_dir)/args.mode_num  
                    pred_intention.append(pred_trajectory[i,pred_intention_ids])
                    pred_intention_score.append(cluster_probs) 
                    id_with_modes.append(i) 
            argo_pred[mapping[i]['file_name']] = structs.MultiScoredTrajectory(pred_score[i].copy(), pred_trajectory[i].copy()) 
        
        pred_score = [scipy.special.softmax(pred_score[i]) for i in range(batch_size)]
        eval_instance_argoverse(batch_size, args, pred_trajectory, pred_score, pred_intention,pred_intention_score, mapping, file2pred, file2score, file2pred_int, 
                                            file2score_int, city_name, file2labels, DEs, iter_bar,id_with_modes)

    if args.argoverse:
        from dataset_argoverse import post_eval
        post_eval(args, file2pred, file2pred_int,file2score, file2score_int, file2labels, DEs, city_name, agent_dir_var_list, agent_dir_int_var_list,opposite_dir_batch, max_guesses)


def main():
    parser = argparse.ArgumentParser()
    utils.add_argument(parser)
    args: utils.Args = parser.parse_args()
    utils.init(args, logger)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    logger.info("device: {}".format(device))

    do_eval(args)


if __name__ == "__main__":
    main()
