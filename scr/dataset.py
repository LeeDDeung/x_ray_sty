#src/dataset.py
# ------------------------------------------------------------
# Chest X-ray Dataset (Image Folder 기반)
# ------------------------------------------------------------
#  목표
# - data/train, data/val, data/test 폴더를 읽어 PyTorch Dataset으로 제공
# - 클래스 불균형(NORMAL 1200 / PNEUMONIA 3800)을 고려해
#   -> class_to_idx / counts / weights(선택)까지 제공
# - "데이터 누수(leakage)"를 피하기 위해
#   -> train/val/test를 폴더 단위로 명확히 분리해서 로딩
#
#  핵심 동작(내부 메커니즘)
# - torchvision.datasets.ImageFolder는
#   root_dir 아래의 "하위 폴더 이름"을 클래스(label)로 인식한다.
#   예) data/train/NORMAL, data/train/PNEUMONIA
# - 각 이미지 파일 경로를 (path, class_index) 형태로 리스트로 만든 뒤,
#   __getitem__에서 이미지를 로드하고 transform 적용 후 (image_tensor, label)을 반환한다.
# ------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose

PathLike = Union[str, Path]

@dataclass
class SplitStats:
    """
   한 split(train/val/test)에 대한 통계 정보 묶음.

   한 split(train/val/test)에 대한 통계 정보 묶음.

    - class_to_idx: {"NORMAL": 0, "PNEUMONIA": 1} 같은 매핑
    - idx_to_class: {0: "NORMAL", 1: "PNEUMONIA"} 역매핑
    - class_counts: {0: 1200, 1: 3800} 처럼 클래스별 샘플 수
    - class_weights: CrossEntropyLoss weight로 바로 넣기 좋게 만든 tensor
        * 기본 전략: weight[i] = total / (num_classes * count[i])
          -> 소수 클래스에 더 큰 가중치가 생김
    """
    class_to_idx: Dict[str, int]
    idx_to_class: Dict[int, str]
    class_count: Dict[int, int]
    class_weights: torch.Tensor

class ChestXrayDataset(Dataset):
    """
    Chest X-ray 분류용 Dataset (ImageFolder wrapper)

    반환값(return)
    - __getitem__(idx) -> (image, label)
      image: torch.Tensor, shape = (C, H, W)
      label: int (0또는 1)

    왜 wrapper를 쓰나?
    - ImageFolder만 써도 되지만,
      프로젝트에서 자주 필요한 "통계/가중치/라벨맵"을 함께 제공하려고 wrapper로 감싼다.
    """
    def __init__(
            self,
            root_dir:PathLike,
            transform: Optional[Compose] = None,
    ):
        """
         Args:
            root_dir:
                예) "data/train" 또는 "data/val" 또는 "data/test"
                내부에 NORMAL/, PNEUMONIA/ 폴더가 있어야 한다.
            transform:
                torchvision transform pipeline (train용/val용 다르게 넣기)
        """
        self.root_dir = Path(root_dir)
        if not self.root_dir.exists():
            raise FileNotFoundError(f"[ChestXrayDataset]root_dir not found: {self.root_dir}")
        # ImageFolder가 실제로 하는 일:
        # - root_dir 아래 하위 폴더들을 클래스 이름으로 인식
        # - 각 이미지 파일들을 스캔해서 samples 리스트 생성
        #   samples: List[(filepath, class_index)]
        self.ds = ImageFolder(root=str(self.root_dir),transform=transform)

        # class_to_idx 예: {"NORMAL":0, "PNEUMONIA":1}
        self.class_to_idx: Dict[str, int] = dict(self.ds.class_to_idx)
        self.idx_to_class: Dict[int, str] = {v: k for k, v in self.class_to_idx.items()}

        # class_count 계산: label별 샘플 수 집계
        # ds.targets: 각 샘플의 label index 리스트 (len == len(ds))
        self.class_counts: Dict[int, int] = self._count_by_class(self.ds.targets)

        # class_weights: 불균형 완화용 가중치(선택적으로 사용)
        # 기본 전략: total/(K*count)
        self.class_weights: torch.Tensor = self._make_class_weight(self.class_counts)

    def __len__(self) -> int:
        """Dataset의 전체 샘플 수 반환"""
        return len(self.ds)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        idx 번째 샘플 반환

        Returns:
            image:(C, H, W) float tensor
            label: int(class index)

        내부 동작:
        -ImageFolder가 해당 idx의 파일을 읽음(PIL)
        -transform이 있으면 적용
        -label은 하위폴더 기반으로 부여됨 (NOMAL=0, PNEUMONIA=1 등)
        """
        image, label = self.ds[idx]
        return image, label
    
    #-------------------
    # 통계/가중치 유틸
    #-------------------
    @staticmethod
    def _count_by_class(targets: List[int]) -> Dict[int, int]:
        """targets 리스트(라벨들) 를 받아 label별 개수를 세어 dict로 반환"""
        counts: Dict[int, int] = {}
        for t in targets:
            counts[t] = counts.get(t, 0) + 1
        return counts
    @staticmethod
    def _make_class_weights(class_count: Dict[int, int]) -> torch.Tensor:
        