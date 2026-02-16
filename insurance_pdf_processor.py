"""
보험사/금융권 비정형 PDF 분석·정제 파이프라인
==============================================

보험 약관, 계리 보고서, 공시 자료 등 고도로 복잡한 비정형 PDF를
레이아웃 분석 → 텍스트 추출 → 정제 → 중복 제거 → 청킹까지
단일 파이프라인으로 처리하는 프로덕션 수준 모듈.

핵심 기술 스택:
    - PyMuPDF (fitz): 좌표 기반 텍스트 블록 추출
    - RapidOCR (ONNX Runtime): 한국어·영어·수식·숫자 OCR
      - CUDA 서버: CUDAExecutionProvider
      - Mac (Apple Silicon): CoreMLExecutionProvider (GPU/ANE 가속)
      - 폴백: CPUExecutionProvider
    - LangChain RecursiveCharacterTextSplitter: RAG 최적화 청킹
"""

from __future__ import annotations

import hashlib
import io
import logging
import platform
import re
import statistics
import unicodedata
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import fitz  # PyMuPDF
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PIL import Image

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s] %(levelname)-8s %(name)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(_handler)


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------
@dataclass
class BoundingBox:
    """페이지 내 본문 영역 좌표."""

    x0: float
    y0: float  # 상단 경계 (헤더 제외)
    x1: float
    y1: float  # 하단 경계 (푸터 제외)


@dataclass
class TextBlock:
    """좌표 정보가 부착된 텍스트 블록."""

    x0: float
    y0: float
    x1: float
    y1: float
    text: str
    block_type: str = "text"  # "text" | "table"


@dataclass
class ChunkMetadata:
    """RAG 벡터 DB에 함께 저장될 메타데이터."""

    source_file: str
    page_number: int
    bounding_box_coord: dict[str, float]
    extraction_timestamp: str
    chunk_index: int = 0
    sha256: str = ""
    extraction_method: str = "pymupdf"  # "pymupdf" | "rapidocr"


@dataclass
class ProcessedChunk:
    """최종 산출물: 청크 텍스트 + 메타데이터."""

    text: str
    metadata: ChunkMetadata


# ---------------------------------------------------------------------------
# 실행 환경 자동 감지
# ---------------------------------------------------------------------------
def _detect_onnx_providers() -> list[tuple[str, dict] | str]:
    """현재 환경에 최적화된 ONNX Runtime ExecutionProvider 목록을 반환한다.

    감지 우선순위:
        1) NVIDIA CUDA → CUDAExecutionProvider
        2) macOS (Apple Silicon) → CoreMLExecutionProvider
        3) 폴백 → CPUExecutionProvider

    Returns:
        onnxruntime.InferenceSession에 전달할 providers 리스트.
    """
    import onnxruntime as ort

    available = ort.get_available_providers()
    providers: list[tuple[str, dict] | str] = []

    # 1) CUDA
    if "CUDAExecutionProvider" in available:
        providers.append(
            (
                "CUDAExecutionProvider",
                {
                    "device_id": 0,
                    "arena_extend_strategy": "kNextPowerOfTwo",
                    "cudnn_conv_algo_search": "EXHAUSTIVE",
                },
            )
        )
        logger.info("ONNX Runtime: CUDAExecutionProvider 감지")

    # 2) CoreML (macOS)
    if "CoreMLExecutionProvider" in available and platform.system() == "Darwin":
        providers.append(
            (
                "CoreMLExecutionProvider",
                {
                    "ModelFormat": "MLProgram",
                    "MLComputeUnits": "ALL",  # CPU + GPU + ANE
                    "RequireStaticInputShapes": "0",
                },
            )
        )
        logger.info("ONNX Runtime: CoreMLExecutionProvider 감지 (Apple Silicon)")

    # 3) CPU 폴백 (항상 마지막)
    providers.append("CPUExecutionProvider")

    return providers


# ---------------------------------------------------------------------------
# OCR Fallback 추상 인터페이스
# ---------------------------------------------------------------------------
class OCREngineBase(ABC):
    """OCR 엔진의 추상 인터페이스.

    RapidOCR 외에 Tesseract, Google Vision 등으로 교체할 수 있도록
    추상 메서드 기반 설계.
    """

    @abstractmethod
    def extract_text_from_image(
        self, image: Image.Image, lang: str = "korean"
    ) -> list[TextBlock]:
        """PIL Image로부터 텍스트 블록 리스트를 추출한다.

        Args:
            image: PIL Image 객체.
            lang: 인식 대상 언어.

        Returns:
            좌표 정보가 포함된 TextBlock 리스트.
        """
        ...


class RapidOCREngine(OCREngineBase):
    """RapidOCR 기반 한국어·영어·수식·숫자 인식 엔진.

    PaddleOCR의 PP-OCRv4/v5 모델을 ONNX로 변환하여
    onnxruntime으로 구동한다. PaddlePaddle 프레임워크 의존성 없이
    CUDA / CoreML(Mac) / CPU 가속을 지원한다.

    환경별 가속:
        - NVIDIA GPU 서버 → CUDAExecutionProvider (RTX 3090 등)
        - Apple Silicon Mac → CoreMLExecutionProvider (GPU + ANE)
        - 기타 → CPUExecutionProvider

    Attributes:
        _engine: RapidOCR 인스턴스 (lazy init).
        _use_cuda: CUDA 사용 여부.
        _use_coreml: CoreML 사용 여부.
    """

    def __init__(
        self,
        use_cuda: Optional[bool] = None,
        use_coreml: Optional[bool] = None,
    ) -> None:
        """RapidOCREngine을 초기화한다.

        Args:
            use_cuda: CUDA 강제 사용 여부. None이면 자동 감지.
            use_coreml: CoreML 강제 사용 여부. None이면 자동 감지.
        """
        self._engine = None
        self._use_cuda = use_cuda
        self._use_coreml = use_coreml

    def _get_engine(self):
        """RapidOCR 인스턴스를 지연 초기화한다.

        환경을 자동 감지하여 최적의 ExecutionProvider를 선택하고,
        한국어 인식 모델(PP-OCRv4)로 구성한다.

        Returns:
            초기화된 RapidOCR 인스턴스.
        """
        if self._engine is not None:
            return self._engine

        from rapidocr import RapidOCR

        # 환경 자동 감지
        if self._use_cuda is None and self._use_coreml is None:
            import onnxruntime as ort

            available = ort.get_available_providers()
            self._use_cuda = "CUDAExecutionProvider" in available
            self._use_coreml = (
                "CoreMLExecutionProvider" in available
                and platform.system() == "Darwin"
            )

        params: dict[str, Any] = {
            # 한국어 인식 모델
            "Rec.lang_type": "korean",
            "Rec.engine_type": "onnxruntime",
            "Rec.model_type": "mobile",
            "Rec.ocr_version": "PP-OCRv4",
            # 검출 모델 (CJK 공용)
            "Det.engine_type": "onnxruntime",
            "Det.model_type": "mobile",
            "Det.ocr_version": "PP-OCRv4",
            # 방향 분류
            "Cls.engine_type": "onnxruntime",
            # 검출 파라미터 — 보험 문서 최적화
            "Det.thresh": 0.3,
            "Det.box_thresh": 0.5,
            "Det.unclip_ratio": 1.6,
            # 인식 배치
            "Rec.rec_batch_num": 16,
        }

        # GPU 가속 설정
        if self._use_cuda:
            params["EngineConfig.onnxruntime.use_cuda"] = True
            params["EngineConfig.onnxruntime.cuda_ep_cfg.device_id"] = 0
            logger.info("RapidOCR: CUDA 가속 활성화")
        else:
            params["EngineConfig.onnxruntime.use_cuda"] = False
            if self._use_coreml:
                logger.info("RapidOCR: CoreML 가속 활성화 (Apple Silicon)")
            else:
                logger.info("RapidOCR: CPU 모드")

        self._engine = RapidOCR(params=params)

        # CoreML: RapidOCR 네이티브 지원이 없으므로 세션을 수동 교체
        if self._use_coreml and not self._use_cuda:
            self._patch_coreml_sessions()

        logger.info(
            "RapidOCR 엔진 초기화 완료 (lang=korean, ocr_version=PP-OCRv4)"
        )
        return self._engine

    def _patch_coreml_sessions(self) -> None:
        """RapidOCR 내부 ONNX 세션을 CoreMLExecutionProvider로 교체한다.

        RapidOCR v3.x는 CoreML EP를 네이티브로 지원하지 않지만,
        내부 InferenceSession을 사후 교체하여 Apple Silicon
        GPU/ANE 가속을 활용할 수 있다.
        """
        import onnxruntime as ort

        providers = _detect_onnx_providers()

        try:
            # RapidOCR 내부 파이프라인의 각 모듈에서 세션 교체 시도
            pipeline = self._engine
            for attr_name in ("text_det", "text_cls", "text_rec"):
                module = getattr(pipeline, attr_name, None)
                if module is None:
                    continue
                # 추론 엔진 내부의 session 접근
                infer = getattr(module, "infer", None) or getattr(
                    module, "predictor", None
                )
                if infer is None:
                    continue
                session = getattr(infer, "session", None)
                if session is None:
                    continue
                # 기존 세션의 모델 경로로 CoreML 세션 재생성
                model_path = session.get_modelmeta().graph_name
                # 모델 파일 경로 추출 시도
                model_file = getattr(infer, "model_path", None)
                if model_file and Path(model_file).exists():
                    new_session = ort.InferenceSession(
                        str(model_file), providers=providers
                    )
                    infer.session = new_session
                    logger.debug(
                        "CoreML 세션 교체 완료: %s", attr_name
                    )
        except Exception as e:
            logger.warning(
                "CoreML 세션 교체 실패 (CPU 폴백 유지): %s", e
            )

    def extract_text_from_image(
        self, image: Image.Image, lang: str = "korean"
    ) -> list[TextBlock]:
        """PIL Image에서 RapidOCR로 텍스트를 추출한다.

        Args:
            image: PIL Image 객체 (페이지 렌더링 결과).
            lang: 언어 설정 (현재 korean 고정, 영어 동시 인식).

        Returns:
            좌표 정보 포함 TextBlock 리스트.
            인식 실패 시 빈 리스트.
        """
        engine = self._get_engine()
        img_array = np.array(image)

        try:
            result = engine(img_array)
        except Exception as e:
            logger.error("RapidOCR 인식 실패: %s", e)
            return []

        blocks: list[TextBlock] = []

        # RapidOCR v3.x 출력: result.boxes (N,4,2), result.txts, result.scores
        if result.boxes is None or result.txts is None:
            return blocks

        for box, txt, score in zip(result.boxes, result.txts, result.scores):
            if score < 0.5:
                continue

            # 4점 좌표 (N,4,2) → 바운딩 박스 (x0,y0,x1,y1)
            # box: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
            xs = box[:, 0]
            ys = box[:, 1]

            blocks.append(
                TextBlock(
                    x0=float(np.min(xs)),
                    y0=float(np.min(ys)),
                    x1=float(np.max(xs)),
                    y1=float(np.max(ys)),
                    text=txt.strip(),
                )
            )

        return blocks


# ---------------------------------------------------------------------------
# 핵심 프로세서 클래스
# ---------------------------------------------------------------------------
class InsurancePDFProcessor:
    """보험·금융 비정형 PDF 분석·정제·청킹 파이프라인.

    주요 기능:
        1. 레이아웃 지능형 대응 — 통계 기반 헤더/푸터 감지, 좌표 기반 읽기 순서 보존
        2. 데이터 무결성 — NFKC 표준화, 수치 보호 정규식, 지능형 문장 병합
        3. RAG 최적화 — SHA-256 중복 제거, 정보 밀도 필터, OCR 폴백
        4. 메타데이터 — 소스 파일, 페이지 번호, 좌표, 타임스탬프 추적
        5. 청킹 — RecursiveCharacterTextSplitter 연동

    사용법::

        processor = InsurancePDFProcessor()
        chunks = processor.process("약관.pdf")
        for c in chunks:
            print(c.text[:100], c.metadata)

    Args:
        chunk_size: 청크 최대 문자 수.
        chunk_overlap: 청크 간 겹침 문자 수.
        min_chunk_length: 유효 청크 최소 문자 수.
        garbage_symbol_ratio: 가비지 판별 기호 비율 임계값.
        y_tolerance: 같은 행 판별 Y좌표 허용 오차 (pt).
        ocr_engine: OCR 엔진 인스턴스 (None이면 RapidOCR 자동 감지).
        ocr_dpi: OCR 렌더링 DPI.
        table_separator: 표 내부 셀 구분자.
    """

    # 수치·날짜·단위를 보호하기 위한 패턴들 (컴파일 캐시)
    # ------------------------------------------------------------------
    # 보호 대상: 12,345.67 | 3.14% | 100원 | $1,000 | USD 500
    #            2024-01-15 | 2024.01.15 | 2024/01/15
    # ------------------------------------------------------------------
    _PROTECTED_NUMBER_PATTERN = re.compile(
        r"[\$\₩]?\d[\d,]*\.?\d*\s*(?:%|원|달러|엔|위안|USD|KRW|JPY|EUR|억|만|천)?"
    )
    _DATE_PATTERN = re.compile(
        r"\d{4}[\-\.\/]\d{1,2}[\-\.\/]\d{1,2}"
    )

    # 수치·날짜를 건드리지 않는 안전한 특수문자 제거 정규식
    # Negative lookahead/lookbehind로 숫자 주변 기호 보호
    _SAFE_CLEAN_PATTERN = re.compile(
        r"(?<![0-9\$\₩])"         # 앞에 숫자/통화기호가 아닌 경우만
        r"[^\w\s\.\,\;\:\!\?\-\–\—\(\)\[\]\{\}\"\'\`\/\%\+\=\<\>\@\#\&\*\₩\$\\]"
        r"(?![0-9%원달러엔위안])"  # 뒤에 숫자/단위가 아닌 경우만
    )

    # 문장 종결 패턴 (한국어 + 영어)
    _SENTENCE_END_PATTERN = re.compile(
        r"[\.。\!\?\:\;]$"          # 문장부호 종결
        r"|다\.$|니다\.$|요\.$"      # 한국어 종결 어미
        r"|함\.$|됨\.$|임\.$"        # 한국어 명사형 종결
        r"|것$|음$|됨$|함$"           # 한국어 명사형 (마침표 없이)
    )

    # 표(Table) 행 패턴 — 탭이나 연속 공백으로 구분된 데이터
    _TABLE_ROW_PATTERN = re.compile(
        r"(?:\S+\s{2,}){2,}"  # 2칸 이상의 공백으로 구분된 항목이 2개 이상
    )

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        min_chunk_length: int = 50,
        garbage_symbol_ratio: float = 0.6,
        y_tolerance: float = 5.0,
        ocr_engine: Optional[OCREngineBase] = None,
        ocr_dpi: int = 300,
        table_separator: str = " | ",
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_length = min_chunk_length
        self.garbage_symbol_ratio = garbage_symbol_ratio
        self.y_tolerance = y_tolerance
        self.ocr_engine: OCREngineBase = ocr_engine or RapidOCREngine()
        self.ocr_dpi = ocr_dpi
        self.table_separator = table_separator

        # 중복 제거용 해시 캐시 (프로세스 라이프사이클 동안 유지)
        self._seen_hashes: set[str] = set()

        # LangChain 텍스트 분할기
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", "다. ", ". ", ", ", " ", ""],
            length_function=len,
        )

        logger.info(
            "InsurancePDFProcessor 초기화: chunk_size=%d, overlap=%d, "
            "min_chunk=%d, y_tol=%.1f, ocr_dpi=%d",
            chunk_size,
            chunk_overlap,
            min_chunk_length,
            y_tolerance,
            ocr_dpi,
        )

    # =======================================================================
    # 1. 레이아웃 지능형 대응 (Layout Intelligence)
    # =======================================================================

    def _detect_content_bounds(
        self, blocks: list[tuple], page_height: float
    ) -> BoundingBox:
        """통계 기반으로 본문 영역(Content Bounding Box)을 동적 감지한다.

        Counter와 statistics.mode를 활용하여 전체 텍스트 블록의 Y좌표 분포를
        분석하고, 헤더·푸터가 위치한 '노이즈 영역'을 판별한다.

        알고리즘:
            1) 모든 블록의 y0(상단), y1(하단)을 수집
            2) 5pt 단위로 버킷화하여 빈도 분석
            3) 상위·하위에서 본문과 단절된 희소 영역을 노이즈로 판별
            4) 본문 영역의 경계를 확정

        Args:
            blocks: fitz get_text("blocks") 결과. 각 원소는
                    (x0, y0, x1, y1, text, block_no, block_type) 형태.
            page_height: 페이지 전체 높이 (pt).

        Returns:
            본문 영역의 BoundingBox.
        """
        if not blocks:
            return BoundingBox(0, 0, 612, page_height)

        # 텍스트 블록만 필터 (block_type == 0)
        text_blocks = [b for b in blocks if b[6] == 0]
        if not text_blocks:
            return BoundingBox(0, 0, 612, page_height)

        # Y좌표 수집 (상단 y0, 하단 y1)
        y_tops: list[float] = [b[1] for b in text_blocks]
        y_bottoms: list[float] = [b[3] for b in text_blocks]

        # 5pt 단위 버킷화
        bucket_size = 5.0
        top_buckets = Counter(int(y / bucket_size) for y in y_tops)
        bottom_buckets = Counter(int(y / bucket_size) for y in y_bottoms)

        # 전체 y좌표 분포 (상단 + 하단)
        all_y = y_tops + y_bottoms
        all_buckets = Counter(int(y / bucket_size) for y in all_y)

        # 모드(최빈값)로 본문 밀집 영역 파악
        if len(all_y) >= 3:
            try:
                mode_bucket = statistics.mode(
                    int(y / bucket_size) for y in all_y
                )
            except statistics.StatisticsError:
                mode_bucket = sorted(
                    all_buckets.items(), key=lambda x: x[1], reverse=True
                )[0][0]
        else:
            mode_bucket = int(statistics.median(all_y) / bucket_size)

        # 본문 밀집 영역의 임계 빈도 (전체 블록 수의 5% 또는 최소 1)
        threshold = max(1, len(text_blocks) * 0.05)

        # 상단 경계: 위에서부터 탐색하여 빈도가 임계값 이상인 첫 버킷
        sorted_buckets = sorted(all_buckets.keys())
        header_boundary = 0.0
        for bucket_idx in sorted_buckets:
            if all_buckets[bucket_idx] >= threshold:
                header_boundary = bucket_idx * bucket_size
                break

        # 하단 경계: 아래에서부터 탐색
        footer_boundary = page_height
        for bucket_idx in reversed(sorted_buckets):
            if all_buckets[bucket_idx] >= threshold:
                footer_boundary = (bucket_idx + 1) * bucket_size
                break

        # X좌표 범위 (전체 텍스트의 min/max)
        x_min = min(b[0] for b in text_blocks)
        x_max = max(b[2] for b in text_blocks)

        bbox = BoundingBox(
            x0=x_min,
            y0=max(0.0, header_boundary - bucket_size),  # 약간의 여유
            x1=x_max,
            y1=min(page_height, footer_boundary + bucket_size),
        )

        logger.debug(
            "본문 영역 감지: y0=%.1f, y1=%.1f (page_h=%.1f, blocks=%d)",
            bbox.y0,
            bbox.y1,
            page_height,
            len(text_blocks),
        )
        return bbox

    def _sort_blocks_reading_order(
        self, blocks: list[TextBlock]
    ) -> list[TextBlock]:
        """텍스트 블록을 인간의 독서 순서로 정렬한다.

        정렬 기준:
            1순위 — Y좌표(행): y_tolerance 이내 → 같은 행
            2순위 — X좌표(열): 왼쪽에서 오른쪽

        이 로직은 다단(multi-column) 레이아웃에서도 표 데이터가
        섞이지 않도록 행 단위 그룹핑을 수행한다.

        Args:
            blocks: 추출된 TextBlock 리스트.

        Returns:
            읽기 순서대로 정렬된 TextBlock 리스트.
        """
        if not blocks:
            return blocks

        # y0 기준 1차 정렬
        sorted_by_y = sorted(blocks, key=lambda b: b.y0)

        # 행(row) 그룹핑: y_tolerance 이내이면 같은 행
        rows: list[list[TextBlock]] = []
        current_row: list[TextBlock] = [sorted_by_y[0]]
        current_y = sorted_by_y[0].y0

        for block in sorted_by_y[1:]:
            if abs(block.y0 - current_y) <= self.y_tolerance:
                current_row.append(block)
            else:
                rows.append(current_row)
                current_row = [block]
                current_y = block.y0

        rows.append(current_row)

        # 각 행 내에서 x0 기준 2차 정렬
        result: list[TextBlock] = []
        for row in rows:
            row.sort(key=lambda b: b.x0)
            result.extend(row)

        return result

    # =======================================================================
    # 2. 텍스트 추출 (PyMuPDF + OCR Fallback)
    # =======================================================================

    def _extract_blocks_pymupdf(
        self, page: fitz.Page, content_bbox: BoundingBox
    ) -> list[TextBlock]:
        """PyMuPDF로 페이지에서 텍스트 블록을 추출한다.

        get_text("blocks")를 사용하여 좌표 정보가 포함된 블록 단위
        텍스트를 추출하고, 본문 영역(content_bbox)에 속하는 블록만 필터링한다.

        Args:
            page: fitz.Page 객체.
            content_bbox: 본문 영역 BoundingBox.

        Returns:
            본문 영역 내 TextBlock 리스트.
        """
        raw_blocks = page.get_text("blocks")
        result: list[TextBlock] = []

        for b in raw_blocks:
            x0, y0, x1, y1, text, block_no, block_type = b[:7]

            # 이미지 블록 스킵 (block_type == 1)
            if block_type != 0:
                continue

            # 본문 영역 필터링
            if y1 < content_bbox.y0 or y0 > content_bbox.y1:
                continue

            text = str(text).strip()
            if not text:
                continue

            # 표 행 여부 판별
            bt = "table" if self._TABLE_ROW_PATTERN.search(text) else "text"

            result.append(
                TextBlock(x0=x0, y0=y0, x1=x1, y1=y1, text=text, block_type=bt)
            )

        return result

    def _extract_blocks_ocr(
        self, page: fitz.Page, content_bbox: BoundingBox
    ) -> list[TextBlock]:
        """OCR 폴백: 페이지를 이미지로 렌더링 후 RapidOCR로 텍스트를 추출한다.

        이미지형 PDF 또는 텍스트 추출이 실패한 페이지에 대해 호출.
        렌더링 DPI는 self.ocr_dpi를 따른다.

        Args:
            page: fitz.Page 객체.
            content_bbox: 본문 영역 BoundingBox.

        Returns:
            OCR로 추출된 TextBlock 리스트.
        """
        # 페이지 → 이미지 렌더링
        zoom = self.ocr_dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.open(io.BytesIO(pix.tobytes("png")))

        # OCR 엔진 호출
        ocr_blocks = self.ocr_engine.extract_text_from_image(img)

        # 좌표를 원본 PDF 좌표계로 역변환 (DPI 보정)
        scale_factor = 72.0 / self.ocr_dpi
        result: list[TextBlock] = []

        for block in ocr_blocks:
            # 좌표 스케일 변환
            scaled = TextBlock(
                x0=block.x0 * scale_factor,
                y0=block.y0 * scale_factor,
                x1=block.x1 * scale_factor,
                y1=block.y1 * scale_factor,
                text=block.text,
                block_type=block.block_type,
            )

            # 본문 영역 필터링
            if scaled.y1 < content_bbox.y0 or scaled.y0 > content_bbox.y1:
                continue

            if scaled.text.strip():
                result.append(scaled)

        return result

    # =======================================================================
    # 3. 데이터 무결성 및 정제 (Data Integrity & Cleaning)
    # =======================================================================

    def _normalize_unicode(self, text: str) -> str:
        """NFKC 유니코드 정규화를 수행한다.

        금융 문서 특유의 결합 문자(합자), 전각 기호, 호환용 한자 등을
        표준 코드포인트로 변환한다.

        예시:
            ﬁ → fi, ％ → %, ＄ → $, ㈜ → (주)

        Args:
            text: 원본 텍스트.

        Returns:
            NFKC 정규화된 텍스트.
        """
        return unicodedata.normalize("NFKC", text)

    def _clean_text_safe(self, text: str) -> str:
        """수치 데이터를 보호하면서 불필요한 특수문자를 제거한다.

        핵심 원칙: 보험 계리 데이터의 숫자, 소수점, 통화 기호, 단위,
        날짜가 정규식 정제 과정에서 절대 훼손되지 않아야 한다.

        보호 대상:
            - 숫자·소수점·천단위 쉼표: 12,345.67
            - 통화 기호 + 숫자: $1,000 / ₩50,000
            - 단위: 100원, 3.5%, USD 500
            - 날짜: 2024-01-15, 2024.01.15

        Args:
            text: NFKC 정규화 후 텍스트.

        Returns:
            수치 안전 정제된 텍스트.
        """
        # 1단계: 수치·날짜 영역을 플레이스홀더로 치환하여 보호
        protected: list[tuple[str, str]] = []
        placeholder_idx = 0

        def _protect(match: re.Match) -> str:
            nonlocal placeholder_idx
            token = f"__PROTECTED_{placeholder_idx}__"
            protected.append((token, match.group()))
            placeholder_idx += 1
            return token

        # 날짜 먼저 보호 (숫자 패턴보다 구체적)
        text = self._DATE_PATTERN.sub(_protect, text)
        # 숫자·통화·단위 보호
        text = self._PROTECTED_NUMBER_PATTERN.sub(_protect, text)

        # 2단계: 안전한 특수문자 제거
        text = self._SAFE_CLEAN_PATTERN.sub("", text)

        # 3단계: 연속 공백 정리
        text = re.sub(r"[ \t]+", " ", text)

        # 4단계: 플레이스홀더 복원
        for token, original in protected:
            text = text.replace(token, original)

        return text.strip()

    def _merge_lines_intelligently(self, text: str) -> str:
        """PDF 추출 시 발생하는 강제 줄바꿈을 지능적으로 병합한다.

        규칙:
            - 문장 종결 기호(.!? 등) 또는 한국어 종결 어미로 끝나면 → 줄바꿈 유지
            - 표(Table) 내부 줄바꿈 → table_separator로 보존
            - 그 외 → 공백으로 이어 붙임 (강제 줄바꿈 해소)

        Args:
            text: 정제된 텍스트.

        Returns:
            줄바꿈이 지능적으로 병합된 텍스트.
        """
        lines = text.split("\n")
        if len(lines) <= 1:
            return text

        merged: list[str] = []
        i = 0

        while i < len(lines):
            line = lines[i].strip()

            if not line:
                # 빈 줄은 문단 구분으로 유지
                merged.append("")
                i += 1
                continue

            # 표 행인 경우 별도 구분자로 보존
            if self._TABLE_ROW_PATTERN.search(line):
                merged.append(line)
                i += 1
                continue

            # 문장 종결 판별
            if self._SENTENCE_END_PATTERN.search(line):
                merged.append(line)
                i += 1
                continue

            # 종결되지 않은 행 → 다음 행과 병합 시도
            combined = line
            i += 1
            while i < len(lines):
                next_line = lines[i].strip()
                if not next_line:
                    break
                # 다음 행이 표 행이면 병합 중단
                if self._TABLE_ROW_PATTERN.search(next_line):
                    break
                combined += " " + next_line
                i += 1
                # 병합 후 문장 종결이면 중단
                if self._SENTENCE_END_PATTERN.search(next_line):
                    break

            merged.append(combined)

        return "\n".join(merged)

    def _process_table_blocks(self, blocks: list[TextBlock]) -> list[TextBlock]:
        """표(Table) 블록 내부의 줄바꿈을 구분자로 치환한다.

        Args:
            blocks: TextBlock 리스트.

        Returns:
            표 블록의 줄바꿈이 table_separator로 변환된 TextBlock 리스트.
        """
        result: list[TextBlock] = []
        for block in blocks:
            if block.block_type == "table":
                # 표 내부 줄바꿈 → 구분자
                cleaned_text = block.text.replace(
                    "\n", self.table_separator
                )
                result.append(
                    TextBlock(
                        x0=block.x0,
                        y0=block.y0,
                        x1=block.x1,
                        y1=block.y1,
                        text=cleaned_text,
                        block_type="table",
                    )
                )
            else:
                result.append(block)
        return result

    # =======================================================================
    # 4. RAG 최적화 및 중복 제거
    # =======================================================================

    def _compute_sha256(self, text: str) -> str:
        """텍스트의 SHA-256 해시를 계산한다.

        공백·줄바꿈을 정규화한 후 해시하여, 미세한 포맷 차이로
        동일 내용이 중복 저장되는 것을 방지한다.

        Args:
            text: 해시 대상 텍스트.

        Returns:
            SHA-256 hex digest.
        """
        normalized = re.sub(r"\s+", " ", text.strip().lower())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def _is_duplicate(self, sha256: str) -> bool:
        """해시 기반 중복 여부를 판별한다.

        법적 고지, 워터마크, 회사 정보 등 매 페이지 반복되는
        보일러플레이트가 벡터 DB에 중복 저장되지 않도록 차단한다.

        Args:
            sha256: 청크의 SHA-256 해시.

        Returns:
            True이면 중복 (스킵해야 함).
        """
        if sha256 in self._seen_hashes:
            return True
        self._seen_hashes.add(sha256)
        return False

    def is_valid_chunk(self, text: str) -> bool:
        """청크의 정보 밀도를 검증하여 가비지 여부를 판별한다.

        걸러지는 경우:
            1) 텍스트 길이가 min_chunk_length 미만
            2) 숫자·기호 비중이 garbage_symbol_ratio 초과
               (예: 페이지 번호만 있는 청크, 깨진 인코딩)

        Args:
            text: 검증 대상 텍스트.

        Returns:
            True이면 유효한 청크, False이면 가비지.
        """
        text = text.strip()

        # 길이 검증
        if len(text) < self.min_chunk_length:
            logger.debug("가비지 필터: 길이 미달 (%d < %d)", len(text), self.min_chunk_length)
            return False

        # 기호·숫자 비중 검증
        if not text:
            return False

        alpha_count = sum(1 for c in text if c.isalpha())
        total_count = sum(1 for c in text if not c.isspace())

        if total_count == 0:
            return False

        alpha_ratio = alpha_count / total_count
        if alpha_ratio < (1.0 - self.garbage_symbol_ratio):
            logger.debug(
                "가비지 필터: 문자 비율 낮음 (alpha=%.2f, threshold=%.2f)",
                alpha_ratio,
                1.0 - self.garbage_symbol_ratio,
            )
            return False

        return True

    # =======================================================================
    # 5. 페이지 단위 처리
    # =======================================================================

    def _process_single_page(
        self,
        doc: fitz.Document,
        page_num: int,
        source_file: str,
        timestamp: str,
    ) -> list[ProcessedChunk]:
        """단일 페이지를 완전히 처리한다.

        파이프라인:
            1) 전체 블록 좌표 수집 → 본문 영역 감지
            2) PyMuPDF 텍스트 추출 (실패 시 OCR 폴백)
            3) 읽기 순서 정렬
            4) 유니코드 정규화 → 수치 보호 정제 → 문장 병합
            5) SHA-256 중복 제거 + 정보 밀도 필터
            6) RecursiveCharacterTextSplitter 청킹
            7) 메타데이터 부착

        Args:
            doc: 열린 fitz.Document.
            page_num: 0-based 페이지 인덱스.
            source_file: 소스 PDF 파일 경로.
            timestamp: 추출 타임스탬프 (ISO 8601).

        Returns:
            해당 페이지에서 생성된 ProcessedChunk 리스트.
        """
        page = doc[page_num]
        page_height = page.rect.height
        extraction_method = "pymupdf"

        # 1) 전체 블록 수집 → 본문 영역 감지
        raw_blocks = page.get_text("blocks")
        content_bbox = self._detect_content_bounds(raw_blocks, page_height)

        # 2) PyMuPDF 텍스트 추출
        blocks = self._extract_blocks_pymupdf(page, content_bbox)

        # OCR 폴백: 텍스트 블록이 거의 없으면 이미지형 PDF로 판단
        if len(blocks) < 2:
            logger.info(
                "p%d: PyMuPDF 추출 부족 (%d블록), OCR 폴백 수행",
                page_num + 1,
                len(blocks),
            )
            try:
                blocks = self._extract_blocks_ocr(page, content_bbox)
                extraction_method = "rapidocr"
            except Exception as e:
                logger.error("p%d: OCR 폴백 실패: %s", page_num + 1, e)
                if not blocks:
                    return []

        if not blocks:
            logger.warning("p%d: 추출된 텍스트 블록 없음", page_num + 1)
            return []

        # 3) 표 블록 구분자 처리 → 읽기 순서 정렬
        blocks = self._process_table_blocks(blocks)
        blocks = self._sort_blocks_reading_order(blocks)

        # 4) 전체 텍스트 조립 → 정제 파이프라인
        page_text_parts: list[str] = []
        for block in blocks:
            text = block.text
            text = self._normalize_unicode(text)
            text = self._clean_text_safe(text)
            page_text_parts.append(text)

        # 문장 병합 (표가 아닌 일반 텍스트에 대해)
        full_text = "\n".join(page_text_parts)
        full_text = self._merge_lines_intelligently(full_text)

        # 5) 정보 밀도 필터
        if not self.is_valid_chunk(full_text):
            logger.debug("p%d: 페이지 전체가 가비지 판정", page_num + 1)
            return []

        # 6) RecursiveCharacterTextSplitter 청킹
        chunk_texts = self._splitter.split_text(full_text)

        # 7) 중복 제거 + 메타데이터 부착
        chunks: list[ProcessedChunk] = []
        for idx, chunk_text in enumerate(chunk_texts):
            if not self.is_valid_chunk(chunk_text):
                continue

            sha = self._compute_sha256(chunk_text)
            if self._is_duplicate(sha):
                logger.debug(
                    "p%d chunk%d: 중복 감지, 스킵", page_num + 1, idx
                )
                continue

            metadata = ChunkMetadata(
                source_file=source_file,
                page_number=page_num + 1,  # 1-based
                bounding_box_coord={
                    "x0": content_bbox.x0,
                    "y0": content_bbox.y0,
                    "x1": content_bbox.x1,
                    "y1": content_bbox.y1,
                },
                extraction_timestamp=timestamp,
                chunk_index=idx,
                sha256=sha,
                extraction_method=extraction_method,
            )

            chunks.append(ProcessedChunk(text=chunk_text, metadata=metadata))

        logger.info(
            "p%d: %d블록 → %d청크 생성 (%s)",
            page_num + 1,
            len(blocks),
            len(chunks),
            extraction_method,
        )
        return chunks

    # =======================================================================
    # 6. 공개 API
    # =======================================================================

    def process(
        self,
        pdf_path: str | Path,
        page_range: Optional[tuple[int, int]] = None,
    ) -> list[ProcessedChunk]:
        """PDF 파일을 완전히 처리하여 RAG용 청크 리스트를 반환한다.

        전체 파이프라인:
            파일 열기 → 페이지별 처리(병렬 가능) → 중복 제거 → 청크 반환

        대량 파일 처리 시 특정 페이지 오류가 전체 프로세스를 중단시키지
        않도록 페이지 단위 예외 처리를 수행한다.

        Args:
            pdf_path: PDF 파일 경로.
            page_range: 처리할 페이지 범위 (1-based, inclusive).
                        None이면 전체 페이지 처리.
                        예: (1, 10) → 1~10페이지.

        Returns:
            ProcessedChunk 리스트. 각 원소는 텍스트와 메타데이터를 포함.

        Raises:
            FileNotFoundError: PDF 파일이 존재하지 않을 때.
            fitz.FileDataError: PDF 파일이 손상되었을 때.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")

        source_file = str(pdf_path.resolve())
        timestamp = datetime.now(timezone.utc).isoformat()

        logger.info("처리 시작: %s", source_file)

        doc = fitz.open(str(pdf_path))
        total_pages = doc.page_count

        # 페이지 범위 결정
        if page_range:
            start = max(0, page_range[0] - 1)  # 1-based → 0-based
            end = min(total_pages, page_range[1])
            pages = range(start, end)
        else:
            pages = range(total_pages)

        logger.info("총 %d 페이지 중 %d 페이지 처리 예정", total_pages, len(pages))

        all_chunks: list[ProcessedChunk] = []

        for page_num in pages:
            try:
                page_chunks = self._process_single_page(
                    doc, page_num, source_file, timestamp
                )
                all_chunks.extend(page_chunks)
            except Exception as e:
                logger.error(
                    "p%d 처리 중 오류 발생 (계속 진행): %s",
                    page_num + 1,
                    e,
                    exc_info=True,
                )
                continue

        doc.close()

        logger.info(
            "처리 완료: %s → 총 %d 청크 생성",
            pdf_path.name,
            len(all_chunks),
        )
        return all_chunks

    def process_batch(
        self,
        pdf_paths: Sequence[str | Path],
        page_range: Optional[tuple[int, int]] = None,
    ) -> dict[str, list[ProcessedChunk]]:
        """여러 PDF를 일괄 처리한다.

        개별 파일 오류가 전체 배치를 중단시키지 않는다.

        Args:
            pdf_paths: PDF 파일 경로 리스트.
            page_range: 각 파일에 적용할 페이지 범위 (선택).

        Returns:
            {파일경로: [ProcessedChunk, ...]} 딕셔너리.
        """
        results: dict[str, list[ProcessedChunk]] = {}

        for path in pdf_paths:
            try:
                chunks = self.process(path, page_range)
                results[str(path)] = chunks
            except Exception as e:
                logger.error("파일 처리 실패 [%s]: %s", path, e, exc_info=True)
                results[str(path)] = []

        total = sum(len(v) for v in results.values())
        logger.info(
            "배치 처리 완료: %d 파일 → 총 %d 청크", len(pdf_paths), total
        )
        return results

    def to_dicts(
        self, chunks: list[ProcessedChunk]
    ) -> list[dict[str, Any]]:
        """ProcessedChunk 리스트를 딕셔너리 리스트로 변환한다.

        벡터 DB (Chroma, Pinecone, Weaviate 등)에 직접 인서트
        가능한 포맷.

        Args:
            chunks: ProcessedChunk 리스트.

        Returns:
            각 원소가 {"text": ..., "metadata": {...}} 형태인 딕셔너리 리스트.
        """
        return [
            {
                "text": chunk.text,
                "metadata": {
                    "source_file": chunk.metadata.source_file,
                    "page_number": chunk.metadata.page_number,
                    "bounding_box_coord": chunk.metadata.bounding_box_coord,
                    "extraction_timestamp": chunk.metadata.extraction_timestamp,
                    "chunk_index": chunk.metadata.chunk_index,
                    "sha256": chunk.metadata.sha256,
                    "extraction_method": chunk.metadata.extraction_method,
                },
            }
            for chunk in chunks
        ]

    def reset_dedup_cache(self) -> None:
        """중복 제거 해시 캐시를 초기화한다.

        새로운 배치 처리 전, 또는 다른 문서 세트를 처리할 때 호출.
        """
        self._seen_hashes.clear()
        logger.info("중복 제거 캐시 초기화 완료")
