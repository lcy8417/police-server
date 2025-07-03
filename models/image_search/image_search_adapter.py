import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F

from .dataset import ImageFromList
from glob import glob

import numpy as np
import os.path as osp
import os
from tqdm import tqdm
import json

ref_root_path = "static/shoes_images/B"


class ImageSearchAdapter:

    ref_memory = "data/ref_vecs.npy"
    device = "cuda"
    ref_names = list(map(lambda x: x.split(".")[0], os.listdir(ref_root_path)))
    ref_names = sorted(ref_names)

    ref_images = list(map(lambda x: osp.join(ref_root_path, x) + ".png", ref_names))

    
    ref_vecs = np.load(ref_memory) if osp.exists(ref_memory) else None
    image_size = 32

    patterns_info = None

    @classmethod
    def get_ref_vectors(cls, model, image_size=1024):

        if cls.ref_vecs is None:
            cls.ref_loader = DataLoader(
                ImageFromList(Image_paths=cls.ref_images, imsize=image_size),
                batch_size=30,
                shuffle=False,
            )

            cls.ref_vecs = cls.extract_vectors(
                net=model,
                loader=cls.ref_loader,
                device=cls.device,
                img_size=len(cls.ref_images),
                batch_size=30,
            ).numpy()

            # memory에 저장
            np.save(cls.ref_memory, cls.ref_vecs)

    @classmethod
    def calculate_distances(cls, query_image, model):
        if cls.ref_vecs is None:
            cls.get_ref_vectors(model, image_size=cls.image_size)

        query_vector = (
            model.forward_test(query_image).detach().cpu().numpy()
        )  # shape: (1, D)
        ref_vectors = cls.ref_vecs

        dists = np.linalg.norm(ref_vectors - query_vector, axis=1)

        # 정렬된 인덱스
        sort_indices = np.argsort(dists)
        page_data = list(map(lambda idx: cls.ref_names[idx], sort_indices))

        return page_data

    @classmethod
    @torch.no_grad()
    def extract_vectors(
        cls, net, loader, ms=[1], device=torch.device("cuda"), img_size=0, batch_size=1
    ):
        net.eval()
        vecs = torch.zeros(img_size, net.outputdim)
        ms = [1]
        if len(ms) == 1:
            for i, input in enumerate(loader):
                vecs[i * batch_size : (i + 1) * batch_size, :] = (
                    net.forward_test(input.to(device)).cpu().data.squeeze()
                )
                print("\r>>>> {}/{} done...".format(i + 1, len(loader)), end="")
        else:
            for i, input in enumerate(
                tqdm(loader, desc="Processing", total=len(loader))
            ):
                vec = torch.zeros(input.shape[0], net.outputdim)
                for s in ms:
                    if s == 1:
                        input_ = input.clone()
                    else:
                        input_ = F.interpolate(
                            input, scale_factor=s, mode="bilinear", align_corners=False
                        )
                    vec += net.forward_test(input_.to(device)).cpu().data.squeeze()
                vec /= len(ms)
                vecs[i * batch_size : (i + 1) * batch_size, :] = F.normalize(
                    vec, p=2, dim=0
                )
                print("\r>>>> {}/{} done...".format(i + 1, len(loader)), end="")

        return vecs

    @classmethod
    def json_load(cls, items):
        """
        JSON 문자열을 파싱하여 리스트로 변환합니다.
        items: JSON 문자열 또는 리스트
        """

        return {
            "top": json.loads(items[0]) if items[0] else [],
            "mid": json.loads(items[1]) if items[1] else [],
            "bottom": json.loads(items[2]) if items[2] else [],
            "outline": json.loads(items[3]) if items[3] else [],
        }

    # 필수 문양이 포함된 이미지들의 리스트만 솎아내기
    @classmethod
    def essential_patterns_filter(cls, page_images, result, data):
        """
        page_images: 검색된 이미지들의 리스트. 거리가 가까운 1순위부터 정렬되어 있음
        result: DB에서 읽어온 모든 신발의 정보
        data: 현재 이미지에서 등록된 필수 문양들 (top, mid, bottom, outline)
        """

        # DB에서 읽어온 결과를 가공. json 문자열을 파싱하여 딕셔너리로 변환
        db_result = {x[0]: cls.json_load(x[1:]) for x in result}
        data = {
            "top": data.top,
            "mid": data.mid,
            "bottom": data.bottom,
            "outline": data.outline,
        }

        filtered_images = []
        for image_name in page_images:
            if image_name in db_result:
                patterns = db_result[image_name]

                if (
                    # 모든 필수 패턴이 일치하는지 확인
                    cls.is_sublist(data["top"], patterns["top"])
                    and cls.is_sublist(data["mid"], patterns["mid"])
                    and cls.is_sublist(data["bottom"], patterns["bottom"])
                    and cls.is_sublist(data["outline"], patterns["outline"])
                ):
                    filtered_images.append(image_name)

        return filtered_images

    @classmethod
    def is_sublist(cls, sub, full):
        if not sub:
            return True
        return all(item in full for item in sub)

    @classmethod
    def load_patterns_info(cls):
        #### 2. 📦 전체 신발 문양 정보 조회 (DB에서 정보 가져오기)
        # TODO 추후 캐싱 필요: 신발이 업데이트 될 때만 로드되도록 최적화 필요

        if cls.patterns_info is not None:
            return cls.patterns_info

        from db.database import direct_get_conn
        from sqlalchemy import text
        from fastapi.responses import JSONResponse
        from fastapi.exceptions import HTTPException
        from fastapi import status

        query = """
        SELECT model_number, top, mid, bottom, outline FROM shoes_data
        """

        conn = direct_get_conn()

        try:
            # DB에서 쿼리 실행
            result = conn.execute(text(query)).fetchall()
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="데이터베이스 쿼리 중 오류가 발생했습니다.",
            )

        conn.close()

        if not result:
            return []

        cls.patterns_info = result

        return cls.patterns_info
