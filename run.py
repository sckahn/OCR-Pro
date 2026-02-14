"""
InsurancePDFProcessor 사용 예시
================================

단일 파일 처리, 배치 처리, 벡터 DB 연동 예시를 포함한다.

사용법::

    python run.py <pdf_path> [--pages START END] [--chunk-size 1000]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from insurance_pdf_processor import InsurancePDFProcessor


def main() -> None:
    parser = argparse.ArgumentParser(
        description="보험·금융 비정형 PDF 분석·정제·청킹 파이프라인",
    )
    parser.add_argument("pdf", help="처리할 PDF 파일 경로")
    parser.add_argument(
        "--pages",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help="처리할 페이지 범위 (1-based, inclusive). 예: --pages 1 10",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="청크 최대 문자 수 (기본: 1000)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="청크 겹침 문자 수 (기본: 200)",
    )
    parser.add_argument(
        "--min-chunk",
        type=int,
        default=50,
        help="유효 청크 최소 문자 수 (기본: 50)",
    )
    parser.add_argument(
        "--ocr-dpi",
        type=int,
        default=300,
        help="OCR 렌더링 DPI (기본: 300)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="결과를 JSON으로 저장할 파일 경로 (선택)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="상세 로그 출력",
    )

    args = parser.parse_args()

    # 로깅 레벨 설정
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="[%(asctime)s] %(levelname)-8s %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # PDF 파일 확인
    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"오류: 파일을 찾을 수 없습니다 — {pdf_path}", file=sys.stderr)
        sys.exit(1)

    # 프로세서 초기화
    processor = InsurancePDFProcessor(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        min_chunk_length=args.min_chunk,
        ocr_dpi=args.ocr_dpi,
    )

    # 처리 실행
    page_range = tuple(args.pages) if args.pages else None
    chunks = processor.process(pdf_path, page_range=page_range)

    # 결과 출력
    print(f"\n{'='*60}")
    print(f"처리 완료: {pdf_path.name}")
    print(f"생성된 청크 수: {len(chunks)}")
    print(f"{'='*60}\n")

    for i, chunk in enumerate(chunks):
        print(f"--- Chunk {i+1} (p{chunk.metadata.page_number}) ---")
        preview = chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text
        print(preview)
        print(f"  [method={chunk.metadata.extraction_method}, "
              f"sha256={chunk.metadata.sha256[:12]}...]")
        print()

    # JSON 출력
    if args.output:
        output_path = Path(args.output)
        dicts = processor.to_dicts(chunks)
        output_path.write_text(
            json.dumps(dicts, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"결과 저장: {output_path}")


if __name__ == "__main__":
    main()
