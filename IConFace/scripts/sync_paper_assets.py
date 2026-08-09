#!/usr/bin/env python3
"""Build the IConFace project-page assets from the final paper sources.

The source map is intentionally explicit.  It mirrors the cases selected in
IConface_v1.tex and IConface_supp_v1.tex without trying to parse TeX macros.
Ordinary face panels are converted to high-quality WebP; diagrams and the
localized-detail composites stay lossless PNG.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


REFERENCE_COLUMNS = [
    ("ref1", "Ref₁"),
    ("deg", "LQ"),
    ("dmdnet", "DMDNet"),
    ("refldm", "ReF-LDM"),
    ("instantrestore", "InstantRestore"),
    ("faceme", "FaceMe"),
    ("refstar", "RefSTAR"),
    ("ours", "IConFace"),
    ("gt", "GT"),
]

MAIN_REFERENCE_COLUMNS = [
    ("ref1", "Reference"),
    ("deg", "LQ"),
    ("refldm", "ReF-LDM"),
    ("instantrestore", "InstantRestore"),
    ("faceme", "FaceMe"),
    ("refstar", "RefSTAR"),
    ("ours", "IConFace"),
    ("gt", "GT"),
]

BLIND_COLUMNS = [
    ("deg", "LQ"),
    ("codeformer", "CodeFormer"),
    ("gfpgan", "GFP-GAN"),
    ("vqfr", "VQFR"),
    ("restoreformerpp", "RF++"),
    ("daefr", "DAEFR"),
    ("ours", "IConFace"),
]

ABLATION_COLUMNS = [
    ("ref1", "Ref₁"),
    ("deg", "LQ"),
    ("concat", "Concat"),
    ("struct", "Concat + Struct"),
    ("id", "Concat + ID"),
    ("1r", "ID + Struct (1R)"),
    ("full", "Full (2R)"),
    ("gt", "GT"),
]

SUPP_REFERENCE_CASES = {
    "CelebA-Test-Ref": [
        "18646", "20257", "26060", "02060", "24093", "00218", "27315", "18375", "08806"
    ],
    "FFHQ-Ref Moderate": [
        "57244", "31223", "48700", "08081", "44570", "55156", "54575", "02647", "15714"
    ],
    "FFHQ-Ref Severe": [
        "02293", "48700", "04716", "12051", "02647", "13427", "14836", "27211", "09880"
    ],
}

SUPP_REFERENCE_DIRS = {
    "CelebA-Test-Ref": "CelebA-Test-Ref",
    "FFHQ-Ref Moderate": "FFHQ-Ref-Moderate",
    "FFHQ-Ref Severe": "FFHQ-Ref-Severe",
}

SUPP_BLIND_CASES = {
    "CelebA-Test": [
        "00000216", "00002645", "00000893", "00000894", "00001737", "00000032", "00000060", "00002781", "00002191"
    ],
    "LFW": [
        "Adrian_Annus_0001_00", "Ben_Davis_0001_00", "Thomas_Haeggstroem_0001_00",
        "Brian_Scalabrine_0001_00", "Abdul_Majeed_Shobokshi_0001_00", "Curtis_Joseph_0001_00",
        "Alfonso_Portillo_0001_00", "Adrien_Brody_0001_00", "AJ_Lamas_0001_00",
    ],
    "CelebChild": [
        "Child__040_Zooey_Deschanel_00", "Adult__005_Chloe_Grace_Moretz_01", "Child__107_John_Wayne_00",
        "Child__061_Demi_Moore_00", "Adult__034_Lady_Gaga_01", "Adult__012_Jackie_Chan_01",
        "Adult__007_Benedict_Cumberbatch_01", "Child__001_Ryan_Gosling_00", "Adult__000_Adele_01",
    ],
    "WebPhoto": [
        "00022_00", "00105_00", "00018_01", "00030_00", "00010_02", "00059_00", "00000_00", "00006_04", "00009_00"
    ],
    "Wider-Test": ["0060", "0000", "0001", "0038", "0039", "0049", "0011", "0020", "0029"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--paper-root",
        type=Path,
        default=Path.home() / "code/local/flux-restoration/paper_submission_iconface",
        help="Directory containing IConface_v1.tex and its figures/fingers folders.",
    )
    parser.add_argument("--quality", type=int, default=92, help="WebP quality for face panels.")
    return parser.parse_args()


def find_image(directory: Path, stem: str) -> Path:
    for suffix in (".png", ".jpg", ".jpeg", ".webp"):
        candidate = directory / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Missing image: {directory}/{stem}.[png|jpg|jpeg|webp]")


def convert_webp(source: Path, destination: Path, quality: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["cwebp", "-quiet", "-q", str(quality), str(source), "-o", str(destination)],
        check=True,
    )


def web_path(path: Path, site_root: Path) -> str:
    return path.relative_to(site_root).as_posix()


def export_case(
    *,
    source_dir: Path,
    sample_id: str,
    columns: list[tuple[str, str]],
    destination_dir: Path,
    site_root: Path,
    quality: int,
    score_values: dict[str, float] | None = None,
) -> dict:
    images: dict[str, str] = {}
    for key, _ in columns:
        source = find_image(source_dir, f"{sample_id}__{key}")
        destination = destination_dir / sample_id / f"{key}.webp"
        convert_webp(source, destination, quality)
        images[key] = web_path(destination, site_root)

    item: dict = {"sample_id": sample_id, "images": images}
    if score_values:
        item["scores"] = {
            key: {"label": "Arc", "value": value}
            for key, value in score_values.items()
        }
    return item


def dataset(
    *,
    dataset_id: str,
    title: str,
    description: str,
    columns: list[tuple[str, str]],
    cases: list[dict],
    open_by_default: bool = False,
) -> dict:
    return {
        "id": dataset_id,
        "title": title,
        "description": description,
        "columns": [{"key": key, "label": label} for key, label in columns],
        "open_by_default": open_by_default,
        "cases": cases,
    }


def make_social_preview(paper_root: Path, destination: Path) -> None:
    teaser = paper_root / "figures/teaser_ffhq_19802"
    names = ["ref1_512", "ref2_512", "ref3_512", "deg_512", "codeformer_512", "refstar_512", "ours_512", "gt_512"]
    sources = [find_image(teaser, name) for name in names]
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    for source in sources:
        command.extend(["-i", str(source)])
    filters = []
    for index in range(len(sources)):
        filters.append(
            f"[{index}:v]scale=300:300:force_original_aspect_ratio=increase,crop=300:300[p{index}]"
        )
    layout = "|".join(
        ["0_0", "300_0", "600_0", "900_0", "0_300", "300_300", "600_300", "900_300"]
    )
    filters.append(
        "".join(f"[p{i}]" for i in range(8))
        + f"xstack=inputs=8:layout={layout},pad=1200:630:0:15:color=white[out]"
    )
    command.extend(["-filter_complex", ";".join(filters), "-map", "[out]", "-q:v", "2", str(destination)])
    subprocess.run(command, check=True)


def make_square_preview(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
            "-vf", "scale=512:512:force_original_aspect_ratio=increase,crop=512:512", "-q:v", "2", str(destination),
        ],
        check=True,
    )


def main() -> None:
    args = parse_args()
    paper_root = args.paper_root.expanduser().resolve()
    site_root = Path(__file__).resolve().parents[1]
    output_root = site_root / "static/gallery/v2"
    image_root = site_root / "static/images/v2"

    for required in (paper_root / "IConface_v1.tex", paper_root / "IConface_supp_v1.tex"):
        if not required.is_file():
            raise FileNotFoundError(required)

    shutil.rmtree(output_root, ignore_errors=True)
    shutil.rmtree(image_root, ignore_errors=True)
    output_root.mkdir(parents=True, exist_ok=True)
    image_root.mkdir(parents=True, exist_ok=True)

    # Teaser and its diagnostic crops.
    teaser_source = paper_root / "figures/teaser_ffhq_19802"
    teaser_map = {
        "ref1": "ref1_512", "ref2": "ref2_512", "ref3": "ref3_512", "deg": "deg_512",
        "codeformer": "codeformer_512", "refstar": "refstar_512", "ours": "ours_512", "gt": "gt_512",
        "codeformer_detail": "codeformer_detail_sq", "refstar_detail": "refstar_detail_sq",
        "ours_detail": "ours_detail_sq", "gt_detail": "gt_detail_sq",
        "codeformer_eye": "codeformer_eye_sq", "refstar_eye": "refstar_eye_sq",
        "ours_eye": "ours_eye_sq", "gt_eye": "gt_eye_sq",
    }
    for destination_name, source_stem in teaser_map.items():
        convert_webp(
            find_image(teaser_source, source_stem),
            image_root / "teaser" / f"{destination_name}.webp",
            args.quality,
        )

    # Diagrams and human-audited detail figures remain lossless.
    lossless_assets = {
        paper_root / "figures/framework_ffhs_22523/framework.png": image_root / "framework.png",
        paper_root / "figures/visible_feature_preservation_v1/pid167_preservation_rate.png": image_root / "pid167_preservation_rate.png",
        paper_root / "figures/visible_feature_preservation_v1/visible_feature_main_case_cropped.png": image_root / "localized_detail_main.png",
        paper_root / "figures/visible_feature_preservation_v1/visible_feature_supp_cases.png": image_root / "localized_detail_extended.png",
    }
    for source, destination in lossless_assets.items():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    groups: list[dict] = []

    # Main-paper reference-aware cases.
    main_ref_root = paper_root / "fingers/iconface_qualitative/finger_512"
    main_ref_specs = [
        ("CelebA-Test-Ref", "CelebA-Test-Ref", "00645"),
        ("FFHQ-Ref Moderate", "FFHQ-Ref-Moderate", "03016"),
        ("FFHQ-Ref Severe", "FFHQ-Ref-Severe", "12744"),
    ]
    main_ref_datasets = []
    for title, folder, case_id in main_ref_specs:
        case = export_case(
            source_dir=main_ref_root / folder,
            sample_id=case_id,
            columns=MAIN_REFERENCE_COLUMNS,
            destination_dir=output_root / "reference-featured" / folder,
            site_root=site_root,
            quality=args.quality,
        )
        main_ref_datasets.append(dataset(
            dataset_id=f"main-ref-{folder.lower()}", title=title,
            description=f"Main-paper comparison, case {case_id}.",
            columns=MAIN_REFERENCE_COLUMNS, cases=[case], open_by_default=True,
        ))
    groups.append({
        "id": "reference-featured",
        "title": "Featured comparisons",
        "description": "Three main-paper cases under the official reference configuration of each method.",
        "datasets": main_ref_datasets,
    })

    # Supplementary reference-aware galleries.
    hard_root = paper_root / "fingers/iconface_hard20"
    extended_ref_datasets = []
    for title, sample_ids in SUPP_REFERENCE_CASES.items():
        folder = SUPP_REFERENCE_DIRS[title]
        cases = [
            export_case(
                source_dir=hard_root / folder,
                sample_id=sample_id,
                columns=REFERENCE_COLUMNS,
                destination_dir=output_root / "reference-extended" / folder,
                site_root=site_root,
                quality=args.quality,
            )
            for sample_id in sample_ids
        ]
        extended_ref_datasets.append(dataset(
            dataset_id=f"supp-ref-{folder.lower()}", title=title,
            description=f"Nine extended {title} comparisons from the supplementary material.",
            columns=REFERENCE_COLUMNS, cases=cases,
        ))
    groups.append({
        "id": "reference-extended",
        "title": "Extended reference-aware gallery",
        "description": "Twenty-seven supplementary cases; each dataset expands only when requested.",
        "datasets": extended_ref_datasets,
    })

    # Main-paper no-reference cases.
    blind_root = paper_root / "fingers/iconface_qualitative/finger_512"
    main_blind_specs = [
        ("LFW", "Adrian_Annus_0001_00"),
        ("CelebChild", "Child__040_Zooey_Deschanel_00"),
        ("WebPhoto", "00022_00"),
        ("Wider-Test", "0060"),
        ("CelebA-Test", "00000216"),
    ]
    main_blind_datasets = []
    for folder, case_id in main_blind_specs:
        case = export_case(
            source_dir=blind_root / folder,
            sample_id=case_id,
            columns=BLIND_COLUMNS,
            destination_dir=output_root / "blind-featured" / folder,
            site_root=site_root,
            quality=args.quality,
        )
        main_blind_datasets.append(dataset(
            dataset_id=f"main-blind-{folder.lower()}", title=folder,
            description=f"Main-paper no-reference comparison, case {case_id}.",
            columns=BLIND_COLUMNS, cases=[case], open_by_default=True,
        ))
    groups.append({
        "id": "blind-featured",
        "title": "Featured comparisons",
        "description": "One main-paper example from each of the five blind-restoration benchmarks.",
        "datasets": main_blind_datasets,
    })

    # Supplementary no-reference galleries.  Some selected cases live in the
    # candidate folder; select the first complete source directory.
    candidate_blind_root = paper_root / "fingers/candidates_100/no_reference"
    extended_blind_datasets = []
    for folder, sample_ids in SUPP_BLIND_CASES.items():
        cases = []
        for sample_id in sample_ids:
            possible_roots = [blind_root / folder, candidate_blind_root / folder]
            source_dir = next(
                (root for root in possible_roots if any((root / f"{sample_id}__deg{s}").is_file() for s in (".png", ".jpg"))),
                None,
            )
            if source_dir is None:
                raise FileNotFoundError(f"No source directory for {folder}/{sample_id}")
            cases.append(export_case(
                source_dir=source_dir,
                sample_id=sample_id,
                columns=BLIND_COLUMNS,
                destination_dir=output_root / "blind-extended" / folder,
                site_root=site_root,
                quality=args.quality,
            ))
        extended_blind_datasets.append(dataset(
            dataset_id=f"supp-blind-{folder.lower()}", title=folder,
            description=f"Nine extended {folder} comparisons from the supplementary material.",
            columns=BLIND_COLUMNS, cases=cases,
        ))
    groups.append({
        "id": "blind-extended",
        "title": "Extended no-reference gallery",
        "description": "Forty-five supplementary cases across all five benchmarks; datasets load on demand.",
        "datasets": extended_blind_datasets,
    })

    # Keep the ablation concise: the two cases selected in the main paper only.
    ablation_root = paper_root / "fingers/candidates_100/ablation"
    ablation_specs = [
        (
            "FFHQ-Ref Moderate", "FFHQ-Ref-Moderate", "03234",
            {"ref1": 1.000, "deg": 0.479, "concat": 0.553, "struct": 0.602, "id": 0.643, "1r": 0.648, "full": 0.656, "gt": 0.607},
        ),
        (
            "FFHQ-Ref Severe", "FFHQ-Ref-Severe", "31223",
            {"ref1": 1.000, "deg": 0.047, "concat": 0.467, "struct": 0.523, "id": 0.535, "1r": 0.540, "full": 0.588, "gt": 0.545},
        ),
    ]
    ablation_datasets = []
    for title, folder, sample_id, scores in ablation_specs:
        case = export_case(
            source_dir=ablation_root / folder,
            sample_id=sample_id,
            columns=ABLATION_COLUMNS,
            destination_dir=output_root / "ablation-main" / folder,
            site_root=site_root,
            quality=args.quality,
            score_values=scores,
        )
        ablation_datasets.append(dataset(
            dataset_id=f"ablation-{folder.lower()}", title=title,
            description=f"Main-paper component comparison, case {sample_id}. Arc labels are cosine similarity to the displayed Ref₁.",
            columns=ABLATION_COLUMNS, cases=[case], open_by_default=True,
        ))
    groups.append({
        "id": "ablation-main",
        "title": "Two representative component comparisons",
        "description": "The concise ablation view mirrors the two cases in the main paper.",
        "datasets": ablation_datasets,
    })

    manifest = {
        "version": 2,
        "source": "IConface_v1.tex and IConface_supp_v1.tex",
        "groups": groups,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    social_preview = site_root / "static/images/social_preview.jpg"
    make_social_preview(paper_root, social_preview)
    square_source = find_image(teaser_source, "ours_512")
    make_square_preview(square_source, site_root.parent / "images/projects/iconface.jpg")
    make_square_preview(square_source, site_root.parent / "images/publications/iconface.jpg")

    print(f"Wrote {manifest_path}")
    print(f"Paper source: {paper_root}")
    print(f"Gallery groups: {len(groups)}")


if __name__ == "__main__":
    main()
