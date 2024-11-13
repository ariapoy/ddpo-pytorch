from importlib import resources
import os
import functools
import random
import inflect
from datasets import load_dataset

IE = inflect.engine()
ASSETS_PATH = resources.files("ddpo_pytorch.assets")


@functools.cache
def _load_lines(path):
    """
    Load lines from a file. First tries to load from `path` directly, and if that doesn't exist, searches the
    `ddpo_pytorch/assets` directory for a file named `path`.
    """
    if not os.path.exists(path):
        newpath = ASSETS_PATH.joinpath(path)
    if not os.path.exists(newpath):
        raise FileNotFoundError(f"Could not find {path} or ddpo_pytorch.assets/{path}")
    path = newpath
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines()]


def from_file(path, low=None, high=None):
    prompts = _load_lines(path)[low:high]
    return random.choice(prompts), {}


def imagenet_all():
    return from_file("imagenet_classes.txt")


def imagenet_animals():
    return from_file("imagenet_classes.txt", 0, 398)


def imagenet_dogs():
    return from_file("imagenet_classes.txt", 151, 269)


def simple_animals():
    return from_file("simple_animals.txt")


def nouns_activities(nouns_file, activities_file):
    nouns = _load_lines(nouns_file)
    activities = _load_lines(activities_file)
    return f"{IE.a(random.choice(nouns))} {random.choice(activities)}", {}

def pickapic():
    if os.path.exists('ddpo_pytorch/assets/pickapic_train.txt'):
        pass
    else:
        dataset_name = 'yuvalkirstain/pickapic_v2'
        dataset_config_name = None
        cache_dir = '/tmp2/lupoy/study-HF/data/pickapics/pickapic_v2/'
        train_data_dir = None
        dataset = load_dataset(
            dataset_name,
            dataset_config_name,
            cache_dir=cache_dir,
            data_dir=train_data_dir,
        )
        prompts = dataset['train']['caption']
        with open('pickapic_train.txt', 'w') as f:
            for prompt in prompts:
                f.write(prompt + '\n')
    return from_file("pickapic_train.txt")

def counting(nouns_file, low, high):
    nouns = _load_lines(nouns_file)
    number = IE.number_to_words(random.randint(low, high))
    noun = random.choice(nouns)
    plural_noun = IE.plural(noun)
    prompt = f"{number} {plural_noun}"
    metadata = {
        "questions": [
            f"How many {plural_noun} are there in this image?",
            f"What animal is in this image?",
        ],
        "answers": [
            number,
            noun,
        ],
    }
    return prompt, metadata

if __name__ == "__main__":
    for _ in range(10):
        prompt, metadata = counting("simple_animals.txt", 1, 9)
        print(prompt, metadata)