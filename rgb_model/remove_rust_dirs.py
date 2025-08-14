import shutil
import os

for split in ['train', 'val', 'test']:
    path = f'datasets/plantvillage_processed/{split}/Rust'
    if os.path.exists(path):
        shutil.rmtree(path)
        print(f'Removed {split}/Rust')
    else:
        print(f'{split}/Rust already removed')

print("\nDone! Rust directories removed from all splits.")