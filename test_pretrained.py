import torch
from pathlib import Path
from dataset import Video
import json
from dataset_test import Video
import sys

def test_pretrained_model(pretrained_path,video_path):

    my_model = torch.load(pretrained_path)
    #my_model.eval()

    from dataset_import import get_training_set, get_validation_set, get_test_set
    from spatial_transforms2 import (
        Compose, Normalize, Scale, CenterCrop, CornerCrop, MultiScaleCornerCrop,
        MultiScaleRandomCrop, RandomHorizontalFlip, ToTensor)
    from temporal_transforms2 import LoopPadding, TemporalRandomCrop
    from target_transforms import ClassLabel, VideoID
    from target_transforms import Compose as TargetCompose
    import os

    norm_value=255 #for rgb data

    scale_step=0.84089 #for the kinetics dataset
    scales = [1]
    n_scales=5
    for i in range(1, n_scales):
        scales.append(scales[-1] * scale_step)
        
    sample_size=112 # default for kinetics
    sample_duration=4 # my choosen window size
    norm_method = Normalize([110.636/norm_value, 103.1606/norm_value, 96.29/norm_value], 
                            [38.756/norm_value, 37.8824/norm_value, 40.03/norm_value]) #per the averages of the dataset
    crop_method = MultiScaleRandomCrop(scales, sample_size)
    spatial_transform = Compose([
                crop_method,
                RandomHorizontalFlip(),
        
                ToTensor(norm_value), norm_method])

    temporal_transform = TemporalRandomCrop(sample_duration)
    target_transform = ClassLabel()

    spatial_transform = Compose([
                ToTensor(norm_value), norm_method])

    spatial_transform = Compose([
                crop_method,
                RandomHorizontalFlip(),
                ToTensor(norm_value), norm_method])

    data = Video(video_path, spatial_transform=spatial_transform,
                    temporal_transform=temporal_transform,
                    sample_duration=1)

    test_loader = torch.utils.data.DataLoader(
        data,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        pin_memory=True)

    class_names=['Nothing','Leaving','Returning']

    correct = 0
    total = 0
    pred_final=[]
    label_final=[]
    video_results=[]
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            labels=labels.cuda()
            outputs = my_model(images)
    #         print(torch.max(outputs, 1))
    #         print(outputs)
            conf, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            predicted=predicted.cuda()
            #print(max(labels), max(predicted)) #for validation
            print(pred_final) #for test (unlabeled)
            #correct += (predicted == labels).sum().item()
    
            predicted=predicted.cpu()
            pred_final.append(max(predicted.data.numpy()))
            #labels=labels.cpu()
            conf=conf.cpu()
            #label_final.append(max(labels.data.numpy()))
            json_label=max(predicted.data.numpy())
            json_label=json_label.tolist()
            json_conf=max(conf.data.numpy())
            json_conf=json_conf.tolist()
            for i in range(3):
                video_results.append({'label': class_names[json_label], 'score': json_conf})

    #print('Accuracy of the network on the test images: %d %%' % (
    #    100 * correct / total))
    # I think there's a better way to print results, look into this

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.plot(pred_final, color='steelblue')
    #ax.plot(label_final, color='red')
    plt.title('Predicted Classes for a Test Set')
    plt.legend(['Predicted Class','True Class'])
    plt.yticks([0, 1, 2],['Nothing','Leaving','Returning'], rotation='horizontal')
    ax.set_ylabel('Predicted Class')
    ax.set_xlabel('Batch (4hz), not time b.c. not all samples are the same length')
    plt.savefig("test_results/test_results.png", dpi=150)

    with open(os.path.join('test_results','test_results.json'),
                'w') as f:
            json.dump(video_results, f)


if __name__=="__main__":
  pretrained_path = sys.argv[1]
  video_path=sys.argv[2]
  test_pretrained_model(pretrained_path,video_path)