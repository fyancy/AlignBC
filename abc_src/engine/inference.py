import os
from collections import defaultdict

import torch
from tqdm import tqdm
import numpy as np

from data.datasets.evaluation import generate_kitti_3d_detection, evaluate_python
from utils.visualize_infer import show_image_with_boxes, show_image_with_boxes_test
from utils.timer import Timer, get_time_str


def inference(model, data_loader, dataset_name, device="cuda",
              results_folder=None, metrics=['R40', 'R11']):
    device = torch.device(device)
    dataset = data_loader.dataset
    print("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
    predict_folder = os.path.join(results_folder, 'data')
    os.makedirs(predict_folder, exist_ok=True)
    # dis_ious = None

    if len(os.listdir(predict_folder)) == 0:
        total_timer = Timer()
        inference_timer = Timer()
        total_timer.tic()
        compute_on_dataset(model, data_loader, device, predict_folder, inference_timer)
        total_time = total_timer.toc()
        total_time_str = get_time_str(total_time)
        print("Total run time: {} ({} s / img)".format(
            total_time_str, total_time / len(dataset))
        )

    print('Finishing generating predictions, start evaluating ...')
    ret_dicts = []
    results = []

    for metric in metrics:
        result, ret_dict = evaluate_python(label_path=dataset.label_dir,
                                           result_path=predict_folder,
                                           label_split_file=dataset.imageset_txt,
                                           current_class=dataset.classes,
                                           metric=metric)

        if os.path.exists(results_folder + fr'/result_{metric}.txt'):
            print(f'Old File exists: {results_folder}/result_{metric}.txt')
            return print(result)
        file = open(results_folder + fr'/result_{metric}.txt', 'w')
        file.write(result)
        file.close()

        ret_dicts.append(ret_dict)
        results.append(result)

    return ret_dicts, results


def compute_on_dataset(model, data_loader, device, predict_folder, timer):
    model.eval()
    cpu_device = torch.device("cpu")
    NMS_TIME = []

    pbar = tqdm(total=len(data_loader.dataset), postfix=dict, mininterval=0.3)
    with torch.no_grad():
        for idx, batch in enumerate(data_loader):
            images, targets, image_ids = batch["images"], batch["targets"], batch["img_ids"]
            images = images.to(device)

            # extract label data for visualize
            # vis_target = targets[0]
            targets = [target.to(device) for target in targets]

            timer.tic()

            # output, time_nms = model(images, targets, nms_time=True)
            # NMS_TIME.append(time_nms)

            output = model(images, targets, nms_time=False)
            # if timer:
            torch.cuda.synchronize()
            timer.toc()

            mean_nms_time = np.mean(NMS_TIME) if len(NMS_TIME) > 0 else 0.
            pbar.set_postfix(**{'total_infer_time/s': timer.total_time, 'FPS': 1. / timer.average_time,
                                'nms_time/img': mean_nms_time})
            pbar.update(1)

            output = output.to(cpu_device)

            # generate txt files for predicted objects
            predict_txt = image_ids[0] + '.txt'
            predict_txt = os.path.join(predict_folder, predict_txt)
            generate_kitti_3d_detection(output, predict_txt)

    pbar.close()
    model.train()
