"""
Benchmark inference with velocity intervention for Qwen-Image-Edit-2509 pipeline.

For each benchmark image, generates outputs at multiple intervention strengths,
allowing systematic comparison of how velocity intervention affects editing.

Uses QwenVelocityAnalyzer for deterministic sampling with intervention support.
"""

import torch
import os
import sys
import json
import argparse
from pathlib import Path
from typing import List
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from VeloEdit import QwenVelocityAnalyzer, get_config
from VeloEdit.core.image_compare import save_compare_artifacts


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark inference with velocity intervention for Qwen (velocity_refactor)"
    )

    # Model
    parser.add_argument("--model-path", type=str, default=None,
                        help="Override model path (default: from qwen_config)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="CUDA device")
    parser.add_argument("--multi-gpu", action="store_true",
                        help="Use device_map='balanced' for multi-GPU")

    # Sampling
    parser.add_argument("--num-inference-steps", type=int, default=None,
                        help="Inference steps (default: from config)")
    parser.add_argument("--guidance-scale", type=float, default=None,
                        help="Guidance scale (default: from config)")
    parser.add_argument("--true-cfg-scale", type=float, default=None,
                        help="True CFG scale for Qwen (default: from config)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    # Velocity intervention
    parser.add_argument("--preserve-intervention-steps", type=int, default=6,
                        help="Intervention steps for preserve/non-edit region; negative values mean effective_steps + value")
    parser.add_argument("--edit-intervention-steps", type=int, default=6,
                        help="Intervention steps for edit region; negative values mean effective_steps + value")
    parser.add_argument("--similarity-threshold", type=float, default=0.8,
                        help="Similarity threshold for intervention")
    parser.add_argument("--similarity-mode", type=str, default="elementwise",
                        choices=["elementwise", "cosine"],
                        help="Similarity mode for mask generation and intervention")
    parser.add_argument("--disable-interv", action="store_true",
                        help="Disable preserve replacement and edit blending")
    parser.add_argument("--blend-weights", type=str, default="0.5",
                        help="Comma-separated list of blend weights to benchmark")

    # Data
    parser.add_argument("--benchmark-path", type=str,
                        default="/hpfs/jerry/RL4Beauty/kontinuouskontext/benchmark",
                        help="Path to benchmark directory")
    parser.add_argument("--output-path", type=str,
                        default="./benchmark_intervention_outputs_qwen",
                        help="Output directory")

    # Batch processing
    parser.add_argument("--start-idx", type=int, default=None,
                        help="Start index (0-indexed)")
    parser.add_argument("--end-idx", type=int, default=None,
                        help="End index (exclusive)")

    # Output options
    parser.add_argument("--save-masks", action="store_true",
                        help="Save similarity mask images per step")
    parser.add_argument("--save-heatmaps", action="store_true",
                        help="Save similarity heatmap images per step")
    parser.add_argument("--save-step-images", action="store_true",
                        help="Save per-step denoising images")
    parser.add_argument("--save-analysis", action="store_true",
                        help="Save JSON/CSV analysis data per image")

    return parser.parse_args()


def parse_blend_weights(weights_str):
    """Parse comma-separated blend weights into sorted float list."""
    return sorted(set(float(w.strip()) for w in weights_str.split(",")))


def strength_to_filename(strength: float) -> str:
    """Convert strength value to filename format: strength_X_XX.jpg"""
    s = f"{strength:.2f}".replace(".", "_")
    return f"strength_{s}.jpg"


def save_compare_if_complete(output_dir: str, blend_weights: List[float]) -> None:
    expected_paths = [
        Path(output_dir) / strength_to_filename(round(1.0 - bw, 2))
        for bw in blend_weights
    ]
    if not all(path.exists() for path in expected_paths):
        missing = [str(path.name) for path in expected_paths if not path.exists()]
        print(f"[Compare] Skip incomplete folder {output_dir}, missing: {missing}")
        return

    saved_paths = save_compare_artifacts(expected_paths, output_dir)
    print(f"[Compare] Saved: {', '.join(saved_paths)}")


def run_benchmark(args, analyzer):
    """Run benchmark with multiple intervention strengths."""
    config = get_config("qwen")

    # Load mapping file
    mapping_path = os.path.join(args.benchmark_path, "mapping_file.json")
    with open(mapping_path, 'r') as f:
        mapping_data = json.load(f)

    image_ids = sorted(mapping_data.keys())
    total = len(image_ids)

    # Batch slicing
    start = args.start_idx if args.start_idx is not None else 0
    end = args.end_idx if args.end_idx is not None else total
    image_ids = image_ids[start:end]
    print(f"[Benchmark] Processing images {start} to {end} ({len(image_ids)} images)")

    # Inference parameters
    num_steps = args.num_inference_steps or config.sampling.num_inference_steps
    seed = args.seed
    preserve_steps = args.preserve_intervention_steps
    edit_steps = args.edit_intervention_steps
    enable_interv = not args.disable_interv
    threshold = args.similarity_threshold
    similarity_mode = args.similarity_mode

    # Parse blend weights to benchmark
    blend_weights_list = parse_blend_weights(args.blend_weights)
    print(f"[Config] Preserve steps: {preserve_steps}")
    print(f"[Config] Edit steps: {edit_steps}")
    print(f"[Config] Blend weights (a values) to benchmark: {blend_weights_list}")
    print(f"[Config] Corresponding strengths (1-a): {[round(1-a, 2) for a in blend_weights_list]}")
    print(
        f"[Config] similarity_threshold={threshold}, "
        f"similarity_mode={similarity_mode}, enable_interv={enable_interv}"
    )

    mode_suffix = "" if similarity_mode == "elementwise" else f"_{similarity_mode}"
    top_dir_name = f"{num_steps}_pres{preserve_steps}_edit{edit_steps}_{threshold}{mode_suffix}"
    top_dir = os.path.join(args.output_path, top_dir_name)
    os.makedirs(top_dir, exist_ok=True)

    print(f"\n[Config] Output: {top_dir}")
    print(f"[Config] Will generate {len(blend_weights_list)} strength files per image")

    for idx, image_id in enumerate(tqdm(image_ids, desc=f"pres={preserve_steps},edit={edit_steps}")):
        entry = mapping_data[image_id]
        image_rel_path = entry['image_path']
        instruction = entry['editing_instruction']

        full_image_path = os.path.join(
            args.benchmark_path, "annotation_images", image_rel_path
        )
        if not os.path.exists(full_image_path):
            print(f"[Warning] Image not found: {full_image_path}")
            continue

        orig_image = Image.open(full_image_path).convert("RGB")

        # Build output directory
        image_path_dir = os.path.dirname(image_rel_path)
        output_dir = os.path.join(top_dir, image_path_dir, image_id)
        os.makedirs(output_dir, exist_ok=True)

        # Save original
        orig_path = os.path.join(output_dir, "original.jpg")
        if not os.path.exists(orig_path):
            orig_image.save(orig_path)

        metadata = {
            "image_id": image_id,
            "original_path": image_rel_path,
            "editing_instruction": instruction,
            "editing_type_id": entry.get("editing_type_id", ""),
            "num_inference_steps": num_steps,
            "seed": seed,
            "preserve_intervention_steps": preserve_steps,
            "edit_intervention_steps": edit_steps,
            "similarity_threshold": threshold,
            "similarity_mode": similarity_mode,
            "enable_interv": enable_interv,
            "model": "Qwen-Image-Edit-2509",
            "strengths": {},
        }

        for bw in blend_weights_list:
            strength = round(1.0 - bw, 2)

            intervention_config = {
                "num_inference_steps": num_steps,
                "seed": seed,
                "preserve_intervention_steps": preserve_steps,
                "edit_intervention_steps": edit_steps,
                "similarity_threshold": threshold,
                "similarity_mode": similarity_mode,
                "enable_interv": enable_interv,
                "blend_weight": bw,
            }

            result = analyzer.analyze(
                image=orig_image,
                prompt=instruction,
                image_path=full_image_path,
                intervention_config=intervention_config,
            )

            if result.generated_image:
                strength_filename = strength_to_filename(strength)
                result.generated_image.save(os.path.join(output_dir, strength_filename))

            metadata["strengths"][f"{strength:.2f}"] = {
                "blend_weight_a": bw,
                "interventions_applied": result.interventions_applied,
                "preserve_interventions_applied": result.preserve_interventions_applied,
                "edit_interventions_applied": result.edit_interventions_applied,
            }

            if args.save_masks and result.similarity_mask_images:
                masks_dir = os.path.join(output_dir, f"masks_s{strength:.2f}")
                os.makedirs(masks_dir, exist_ok=True)
                for i, img in enumerate(result.similarity_mask_images):
                    img.save(os.path.join(masks_dir, f"mask_{i:02d}.png"))

            if args.save_heatmaps and result.similarity_heatmap_images:
                heatmaps_dir = os.path.join(output_dir, f"heatmaps_s{strength:.2f}")
                os.makedirs(heatmaps_dir, exist_ok=True)
                for i, img in enumerate(result.similarity_heatmap_images):
                    img.save(os.path.join(heatmaps_dir, f"heatmap_{i:02d}.png"))

            if args.save_step_images and result.step_images:
                steps_dir = os.path.join(output_dir, f"steps_s{strength:.2f}")
                os.makedirs(steps_dir, exist_ok=True)
                for i, img in enumerate(result.step_images):
                    img.save(os.path.join(steps_dir, f"step_{i:02d}.png"))

            torch.cuda.empty_cache()

        with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)

        if args.save_analysis:
            from VeloEdit.output import export_to_json
            export_to_json(metadata, os.path.join(output_dir, "analysis.json"))

        save_compare_if_complete(output_dir, blend_weights_list)

        if (idx + 1) % 50 == 0:
            print(f"[Progress] {idx + 1}/{len(image_ids)} images processed")

    print(f"\n[Done] Results saved to: {args.output_path}")


def main():
    args = parse_args()
    print(f"[Args] {args}")

    config = get_config("qwen")
    if args.model_path:
        config.model.path = args.model_path
    if args.num_inference_steps:
        config.sampling.num_inference_steps = args.num_inference_steps
    if args.guidance_scale:
        config.sampling.guidance_scale = args.guidance_scale
    if args.true_cfg_scale:
        config.sampling.true_cfg_scale = args.true_cfg_scale

    analyzer = QwenVelocityAnalyzer(
        config=config,
        device=args.device,
        multi_gpu=args.multi_gpu,
    )

    print("[Model] Loading Qwen-Image-Edit-2509 pipeline...")
    analyzer.load_model()
    analyzer.model_loaded = True
    print("[Model] Pipeline ready.")

    run_benchmark(args, analyzer)


if __name__ == "__main__":
    main()
