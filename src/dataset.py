import os
import sys
import torch
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, List, Union
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.audio_processor import AudioProcessor
except ImportError:
    from audio_processor import AudioProcessor


class ASVspoofDataset(Dataset):
    def __init__(
        self,
        protocol_file: Union[str, Path],
        audio_dir: Optional[Union[str, Path]] = None,
        processor: Optional[AudioProcessor] = None,
        augment: bool = False
    ):
        self.protocol_file = Path(protocol_file)
        self.processor = processor if processor is not None else AudioProcessor()
        self.augment = augment
        self.items: List[Tuple[Path, int, str]] = []

        self._parse_protocol(audio_dir)

    def _parse_protocol(self, audio_dir: Optional[Union[str, Path]]):
        if not self.protocol_file.exists():
            raise FileNotFoundError(f"Protocol file not found: {self.protocol_file}")

        df = pd.read_csv(self.protocol_file)
        for _, row in df.iterrows():
            if 'path' in row and pd.notna(row['path']) and Path(row['path']).exists():
                file_path = Path(row['path'])
            elif audio_dir is not None:
                file_path = Path(audio_dir) / f"{row['audio_file']}.wav"
            else:
                file_path = Path("data/sample_audio") / f"{row['audio_file']}.wav"

            target = int(row['target']) if 'target' in row else (0 if row['label'] == 'bonafide' else 1)
            file_id = str(row.get('audio_file', file_path.stem))
            self.items.append((file_path, target, file_id))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        file_path, target, file_id = self.items[idx]

        if not file_path.exists():
            tensor = torch.zeros((2, self.processor.n_mels, 126), dtype=torch.float32)
            return tensor, target, file_id

        processed = self.processor.process(file_path, augment=self.augment)
        return torch.from_numpy(processed["dual_tensor"]), target, file_id


def create_dataloader(
    protocol_file: Union[str, Path],
    audio_dir: Optional[Union[str, Path]] = None,
    batch_size: int = 16,
    shuffle: bool = True,
    augment: bool = False,
    num_workers: int = 0
) -> DataLoader:
    dataset = ASVspoofDataset(protocol_file=protocol_file, audio_dir=audio_dir, augment=augment)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
