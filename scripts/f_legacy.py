from torch.utils.data import Sampler
import random
import torch
class CVSSampler(Sampler):
    def __init__(self, dataset, upsample_factor, reshuffle=True):
        self.dataset = dataset
        self.indices = list(range(len(dataset)))
        self.labels = [tuple(label.item() for label in dataset[idx][1]) for idx in range(len(dataset))]  # Convert labels to tuples
        self.upsample_factor = upsample_factor
        self.reshuffle = reshuffle
        self.upsampled_indices = self.create_upsampled_indices()

    def create_upsampled_indices(self):
        upsampled_indices = []
        label_map = {
            (0., 0., 0.): 0,
            (1., 0., 0.): 1,
            (0., 1., 0.): 2,
            (0., 0., 1.): 3,
            (1., 1., 0.): 4,
            (1., 0., 1.): 5,
            (0., 1., 1.): 6,
            (1., 1., 1.): 7
        }
        for idx, label in zip(self.indices, self.labels):
            label_tuple = tuple(label)
            if label_tuple in label_map:
                scale = self.upsample_factor[label_map[label_tuple]]
                upsampled_indices.extend([idx] * int(scale))
                if random.random() < scale - int(scale):
                    upsampled_indices.append(idx)
            else:
                raise RuntimeError(f"Data label outside of expected value. Label {label_tuple}")
        return upsampled_indices

    def __iter__(self):
        if self.reshuffle:
            random.shuffle(self.upsampled_indices)
        return iter(self.upsampled_indices)

    def __len__(self):
        return len(self.upsampled_indices)

def check_memory():
    # Get the current GPU device
    device = torch.cuda.current_device()
    
    # Get the total and allocated memory
    total_memory = torch.cuda.get_device_properties(device).total_memory
    allocated_memory = torch.cuda.memory_allocated(device)
    reserved_memory = torch.cuda.memory_reserved(device)

    # Print the memory information
    print(f"Total Memory: {total_memory / (1024**2):.2f} MB")
    print(f"Allocated Memory: {allocated_memory / (1024**2):.2f} MB")
    print(f"Reserved Memory: {reserved_memory / (1024**2):.2f} MB\n")