import torch
import torchvision
import pickle
import torchvision.transforms as transforms
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.save_load_attack import load_clean_trained_model
from utils.aggregate_block.dataset_and_transform_generate import SIMPLE_DATASET_FOR_VISUALIZATION, get_transform
from utils.visualize_dataset import visualize_random_samples_from_clean_dataset, visualize_random_samples_from_bd_dataset
import os



def check_and_visualize_saved_dataset(args, file_path):
    with open(file_path, 'rb') as file:
        top_k_saved_images = pickle.load(file)

    top_k_dataset = SIMPLE_DATASET_FOR_VISUALIZATION(top_k_saved_images, target_label=args.attack_target)

    visualize_random_samples_from_clean_dataset(top_k_dataset, "top_k_dataset")

def save_top_k_from_target_label_train(args):
    # Load the pre-trained model
    model = generate_cls_model(
        model_name=args.model,
        num_classes=args.num_classes,
        image_size=args.img_size[0],
    )

    current_directory = os.getcwd()
    print("Current directory:", current_directory)

    clean_trained_model_dict = load_clean_trained_model('./clean_trained_model/record/badnet_0_1/attack_result.pt')

    model.load_state_dict(clean_trained_model_dict['model'])

    device = torch.device(
        (
            f"cuda:{[int(i) for i in args.device[5:].split(',')][0]}" if "," in args.device else args.device
            # since DataParallel only allow .to("cuda")
        ) if torch.cuda.is_available() else "cpu"
    )

    model.to(args.device)
    model.eval()

    if not args.dataset.startswith('test'):
        test_img_transform = get_transform(args.dataset, *(args.img_size[:2]), train=False)
    else:
        # test folder datset, use the mnist transform for convenience
        test_img_transform = get_transform('mnist', *(args.img_size[:2]), train=False)

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=None)

    # Filter out the records with label 0
    label_0_indices = [i for i, (image, label) in enumerate(trainset) if label == 0]

    # Select the top k records with label 0 based on model's prediction
    top_k_label_0_indices = sorted(label_0_indices,
                                   key=lambda x: model(test_img_transform(trainset[x][0]).unsqueeze(0).to(device))[
                                       0, 0].item(), reverse=True)[:args.top_k]

    # Get the corresponding images and labels
    top_k_selected_images = [trainset[i][0] for i in top_k_label_0_indices]

    # File path
    file_path = "../clean_trained_model/top_k_selected_images.pkl"

    # Save list using pickle
    with open(file_path, 'wb') as file:
        pickle.dump(top_k_selected_images, file)

    check_and_visualize_saved_dataset(args, file_path)