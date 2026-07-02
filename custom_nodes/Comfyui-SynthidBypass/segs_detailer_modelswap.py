import logging
import os
import sys
from pathlib import Path

import comfy
from nodes import MAX_RESOLUTION


# Ensure Impact Pack modules are importable even if this node loads first.
_THIS_DIR = Path(__file__).resolve().parent
_IMPACT_MODULES_DIR = None

for _parent in (_THIS_DIR.parent, *_THIS_DIR.parents):
    _candidates = [
        _parent / "ComfyUI-Impact-Pack" / "modules",
        _parent / "comfyui-impact-pack" / "modules",
    ]
    for _candidate in _candidates:
        if _candidate.is_dir():
            _IMPACT_MODULES_DIR = _candidate
            if str(_candidate) not in sys.path:
                sys.path.append(str(_candidate))
            break
    else:
        continue
    break


try:
    from comfy_extras import nodes_differential_diffusion
except Exception:
    nodes_differential_diffusion = None

try:
    from impact import core, utils
    from impact.core import SEG
    _IMPACT_IMPORT_ERROR = None
except Exception as exc:
    core = None
    utils = None
    SEG = None
    _IMPACT_IMPORT_ERROR = exc


class SEGSDetailerModelSwap:
    """
    Track-1 clone of Impact Pack SEGSDetailer that takes direct model/clip/vae/
    positive/negative inputs instead of BASIC_PIPE, so any loaded diffusion
    model path can be wired in directly.

    Adaptive denoise controls are intentionally simplified:
    - adaptive_mode: largest_face or per_face
    - adaptive_ratio: face-ratio that maps to adaptive_denoise_max
    - min/max clamps
    """

    @classmethod
    def INPUT_TYPES(cls):
        schedulers = core.get_schedulers() if core is not None else ["normal"]
        return {
            "required": {
                "image": ("IMAGE",),
                "segs": ("SEGS",),
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "guide_size": ("FLOAT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                "guide_size_for": ("BOOLEAN", {"default": True, "label_on": "bbox", "label_off": "crop_region"}),
                "max_size": ("FLOAT", {"default": 1024, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True}),
                "steps": ("INT", {"default": 8, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (schedulers,),
                "denoise": ("FLOAT", {"default": 0.10, "min": 0.0001, "max": 1.0, "step": 0.01}),
                "noise_mask": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                "force_inpaint": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                "refiner_ratio": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 100}),
                "cycle": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
            },
            "optional": {
                "refiner_model_opt": ("MODEL",),
                "refiner_clip_opt": ("CLIP",),
                "refiner_positive_opt": ("CONDITIONING",),
                "refiner_negative_opt": ("CONDITIONING",),
                "inpaint_model": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                "noise_mask_feather": ("INT", {"default": 20, "min": 0, "max": 100, "step": 1}),
                "adaptive_denoise": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                "adaptive_mode": (("largest_face", "per_face"), {"default": "largest_face"}),
                "adaptive_ratio": ("FLOAT", {"default": 0.10, "min": 0.0001, "max": 1.0, "step": 0.0005}),
                "adaptive_denoise_min": ("FLOAT", {"default": 0.05, "min": 0.0001, "max": 1.0, "step": 0.01}),
                "adaptive_denoise_max": ("FLOAT", {"default": 0.35, "min": 0.0001, "max": 1.0, "step": 0.01}),
                "scheduler_func_opt": ("SCHEDULER_FUNC",),
            },
        }

    RETURN_TYPES = ("SEGS", "IMAGE", "FLOAT", "STRING")
    RETURN_NAMES = ("segs", "cnet_images", "denoise_values", "denoise_report")
    OUTPUT_IS_LIST = (False, True, True, False)
    FUNCTION = "doit"
    CATEGORY = "SynthidBypass"
    DESCRIPTION = (
        "SEGS detailer with direct MODEL/CLIP/VAE/CONDITIONING inputs. "
        "Adaptive denoise: choose largest_face or per_face + one adaptive_ratio."
    )

    @staticmethod
    def _require_impact_pack():
        if core is None or utils is None or SEG is None:
            raise RuntimeError(
                "ComfyUI-Impact-Pack import failed. Install/enable ComfyUI-Impact-Pack first. "
                f"Original error: {_IMPACT_IMPORT_ERROR}"
            )

    @staticmethod
    def _clamp(value, min_value, max_value):
        min_v = float(min_value)
        max_v = float(max_value)
        if max_v < min_v:
            min_v, max_v = max_v, min_v
        return max(min_v, min(max_v, float(value)))

    @staticmethod
    def _compute_bbox_ratio(seg, image):
        try:
            image_h = float(image.shape[1])
            image_w = float(image.shape[2])
            image_area = image_w * image_h
            if image_area <= 0:
                return None

            if seg.bbox is not None and len(seg.bbox) >= 4:
                x1, y1, x2, y2 = seg.bbox[:4]
            elif seg.crop_region is not None and len(seg.crop_region) >= 4:
                x1, y1, x2, y2 = seg.crop_region[:4]
            else:
                return None

            box_w = max(0.0, float(x2) - float(x1))
            box_h = max(0.0, float(y2) - float(y1))
            box_area = box_w * box_h
            if box_area <= 0:
                return None

            return box_area / image_area
        except Exception:
            return None

    @classmethod
    def _compute_scaled_denoise(
        cls,
        base_denoise,
        ratio,
        adaptive_denoise,
        adaptive_ratio,
        adaptive_denoise_min,
        adaptive_denoise_max,
    ):
        base = float(base_denoise)
        min_d = float(adaptive_denoise_min)
        max_d = float(adaptive_denoise_max)
        if max_d < min_d:
            min_d, max_d = max_d, min_d

        # Keep adaptive_denoise arg for backward compatibility with older workflows,
        # but always apply adaptive scaling in the simplified node UX.
        _ = adaptive_denoise

        # ratio == adaptive_ratio -> base denoise
        # ratio above/below scales denoise up/down, then clamp to min/max.
        reference = max(1e-6, float(adaptive_ratio))
        if ratio is None or ratio <= 0.0:
            return cls._clamp(base, min_d, max_d)

        scaled = base * (float(ratio) / reference)
        return cls._clamp(scaled, min_d, max_d)

    @classmethod
    def _compute_largest_ratio(cls, segs, image):
        largest_ratio = None
        largest_seg_idx = None
        for idx, seg in enumerate(segs[1], start=1):
            ratio = cls._compute_bbox_ratio(seg, image)
            if ratio is None:
                continue
            if largest_ratio is None or ratio > largest_ratio:
                largest_ratio = ratio
                largest_seg_idx = idx
        return largest_ratio, largest_seg_idx

    @staticmethod
    def _format_bbox(seg):
        if seg.bbox is None or len(seg.bbox) < 4:
            return "n/a"
        try:
            x1, y1, x2, y2 = seg.bbox[:4]
            return f"{int(x1)},{int(y1)},{int(x2)},{int(y2)}"
        except Exception:
            return "n/a"

    @staticmethod
    def do_detail(
        image,
        segs,
        model,
        clip,
        vae,
        positive,
        negative,
        guide_size,
        guide_size_for,
        max_size,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        noise_mask,
        force_inpaint,
        refiner_ratio,
        batch_size,
        cycle,
        refiner_model_opt=None,
        refiner_clip_opt=None,
        refiner_positive_opt=None,
        refiner_negative_opt=None,
        inpaint_model=False,
        noise_mask_feather=0,
        adaptive_denoise=False,
        adaptive_mode="largest_face",
        adaptive_ratio=0.10,
        adaptive_denoise_min=0.05,
        adaptive_denoise_max=0.35,
        scheduler_func_opt=None,
    ):
        segs = core.segs_scale_match(segs, image.shape)

        new_segs = []
        cnet_pil_list = []
        denoise_values = []
        denoise_report_lines = []

        has_refiner = (
            refiner_model_opt is not None
            and refiner_clip_opt is not None
            and refiner_positive_opt is not None
            and refiner_negative_opt is not None
        )

        refiner_model = refiner_model_opt if has_refiner else None
        refiner_clip = refiner_clip_opt if has_refiner else None
        refiner_positive = refiner_positive_opt if has_refiner else None
        refiner_negative = refiner_negative_opt if has_refiner else None

        if noise_mask_feather > 0 and hasattr(model, "model_options") and "denoise_mask_function" not in model.model_options:
            if nodes_differential_diffusion is not None:
                model = nodes_differential_diffusion.DifferentialDiffusion().execute(model)[0]
            else:
                logging.warning("SEGSDetailerModelSwap: DifferentialDiffusion unavailable; continuing without mask feather optimization.")

        largest_ratio, largest_seg_idx = SEGSDetailerModelSwap._compute_largest_ratio(segs, image)
        largest_face_denoise = SEGSDetailerModelSwap._compute_scaled_denoise(
            denoise,
            largest_ratio,
            adaptive_denoise,
            adaptive_ratio,
            adaptive_denoise_min,
            adaptive_denoise_max,
        )
        largest_ratio_text = "n/a" if largest_ratio is None else f"{largest_ratio:.6f}"
        mode = adaptive_mode if adaptive_mode in ("largest_face", "per_face") else "largest_face"

        for batch_i in range(batch_size):
            seed += 1
            for seg_i, seg in enumerate(segs[1], start=1):
                cropped_image = seg.cropped_image if seg.cropped_image is not None else utils.crop_ndarray4(image.numpy(), seg.crop_region)
                cropped_image = utils.to_tensor(cropped_image)

                if (seg.cropped_mask == 0).all().item():
                    logging.info("SEGSDetailerModelSwap: segment skipped (empty mask)")
                    denoise_report_lines.append(
                        f"batch={batch_i + 1} seg={seg_i} skipped=empty_mask bbox={SEGSDetailerModelSwap._format_bbox(seg)}"
                    )
                    new_segs.append(seg)
                    continue

                cropped_mask = seg.cropped_mask if noise_mask else None

                cropped_positive = [
                    [
                        condition,
                        {
                            k: core.crop_condition_mask(v, image, seg.crop_region) if k == "mask" else v
                            for k, v in details.items()
                        },
                    ]
                    for condition, details in positive
                ]

                cropped_negative = [
                    [
                        condition,
                        {
                            k: core.crop_condition_mask(v, image, seg.crop_region) if k == "mask" else v
                            for k, v in details.items()
                        },
                    ]
                    for condition, details in negative
                ]

                seg_ratio = SEGSDetailerModelSwap._compute_bbox_ratio(seg, image)
                seg_ratio_text = "n/a" if seg_ratio is None else f"{seg_ratio:.6f}"

                per_face_denoise = SEGSDetailerModelSwap._compute_scaled_denoise(
                    denoise,
                    seg_ratio,
                    adaptive_denoise,
                    adaptive_ratio,
                    adaptive_denoise_min,
                    adaptive_denoise_max,
                )

                seg_denoise = per_face_denoise if mode == "per_face" else largest_face_denoise

                denoise_values.append(float(seg_denoise))
                denoise_report_lines.append(
                    f"batch={batch_i + 1} seg={seg_i} base_denoise={float(denoise):.4f} applied_denoise={seg_denoise:.4f} "
                    f"individual_denoise={per_face_denoise:.4f} largest_face_denoise={largest_face_denoise:.4f} "
                    f"mode={mode} seg_ratio={seg_ratio_text} "
                    f"largest_ratio={largest_ratio_text} largest_seg={largest_seg_idx if largest_seg_idx is not None else 'n/a'} "
                    f"adaptive_ratio={float(adaptive_ratio):.6f} bbox={SEGSDetailerModelSwap._format_bbox(seg)}"
                )

                enhanced_image, cnet_pils = core.enhance_detail(
                    cropped_image,
                    model,
                    clip,
                    vae,
                    guide_size,
                    guide_size_for,
                    max_size,
                    seg.bbox,
                    seed,
                    steps,
                    cfg,
                    sampler_name,
                    scheduler,
                    cropped_positive,
                    cropped_negative,
                    seg_denoise,
                    cropped_mask,
                    force_inpaint,
                    refiner_ratio=refiner_ratio,
                    refiner_model=refiner_model,
                    refiner_clip=refiner_clip,
                    refiner_positive=refiner_positive,
                    refiner_negative=refiner_negative,
                    control_net_wrapper=seg.control_net_wrapper,
                    cycle=cycle,
                    inpaint_model=inpaint_model,
                    noise_mask_feather=noise_mask_feather,
                    scheduler_func=scheduler_func_opt,
                )

                if cnet_pils is not None:
                    cnet_pil_list.extend(cnet_pils)

                new_cropped_image = cropped_image if enhanced_image is None else enhanced_image
                new_seg = SEG(
                    utils.to_numpy(new_cropped_image),
                    seg.cropped_mask,
                    seg.confidence,
                    seg.crop_region,
                    seg.bbox,
                    seg.label,
                    None,
                )
                new_segs.append(new_seg)

        if denoise_report_lines:
            denoise_report = "\n".join(denoise_report_lines)
        else:
            denoise_report = "No segments processed."

        return (segs[0], new_segs), cnet_pil_list, denoise_values, denoise_report

    def doit(
        self,
        image,
        segs,
        model,
        clip,
        vae,
        positive,
        negative,
        guide_size,
        guide_size_for,
        max_size,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        noise_mask,
        force_inpaint,
        refiner_ratio,
        batch_size,
        cycle,
        refiner_model_opt=None,
        refiner_clip_opt=None,
        refiner_positive_opt=None,
        refiner_negative_opt=None,
        inpaint_model=False,
        noise_mask_feather=0,
        adaptive_denoise=False,
        adaptive_mode="largest_face",
        adaptive_ratio=0.10,
        adaptive_denoise_min=0.05,
        adaptive_denoise_max=0.35,
        scheduler_func_opt=None,
    ):
        self._require_impact_pack()

        if len(image) > 1:
            raise Exception(
                "[Synthid-Bypass-Facedetailer] image batches are not supported. Use one image at a time."
            )

        out_segs, cnet_pil_list, denoise_values, denoise_report = self.do_detail(
            image,
            segs,
            model,
            clip,
            vae,
            positive,
            negative,
            guide_size,
            guide_size_for,
            max_size,
            seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
            denoise,
            noise_mask,
            force_inpaint,
            refiner_ratio,
            batch_size,
            cycle,
            refiner_model_opt=refiner_model_opt,
            refiner_clip_opt=refiner_clip_opt,
            refiner_positive_opt=refiner_positive_opt,
            refiner_negative_opt=refiner_negative_opt,
            inpaint_model=inpaint_model,
            noise_mask_feather=noise_mask_feather,
            adaptive_denoise=adaptive_denoise,
            adaptive_mode=adaptive_mode,
            adaptive_ratio=adaptive_ratio,
            adaptive_denoise_min=adaptive_denoise_min,
            adaptive_denoise_max=adaptive_denoise_max,
            scheduler_func_opt=scheduler_func_opt,
        )

        if len(cnet_pil_list) == 0:
            cnet_pil_list = [utils.empty_pil_tensor()]

        return out_segs, cnet_pil_list, denoise_values, denoise_report


class SynthidBypassAdaptiveDenoise:
    """
    Resolution-based adaptive denoise calculator.
    No face detection; uses full image dimensions only.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "adaptive_level": ("INT", {"default": 5, "min": 1, "max": 10, "step": 1}),
                "denoise_min": ("FLOAT", {"default": 0.08, "min": 0.0001, "max": 1.0, "step": 0.001}),
                "denoise_max": ("FLOAT", {"default": 0.15, "min": 0.0001, "max": 1.0, "step": 0.001}),
            }
        }

    RETURN_TYPES = ("FLOAT", "INT", "INT", "FLOAT", "STRING")
    RETURN_NAMES = ("denoise", "width", "height", "pixel_ratio", "report")
    FUNCTION = "doit"
    CATEGORY = "SynthidBypass"
    DESCRIPTION = (
        "Computes denoise from image resolution only using adaptive_level (1-10). "
        "Level 5 is neutral calibration."
    )

    @staticmethod
    def _clamp(value, min_value, max_value):
        min_v = float(min_value)
        max_v = float(max_value)
        if max_v < min_v:
            min_v, max_v = max_v, min_v
        return max(min_v, min(max_v, float(value)))

    def doit(self, image, adaptive_level, denoise_min, denoise_max):
        width = int(image.shape[2])
        height = int(image.shape[1])

        # Pixel ratio output is still based on 1MP (1024x1024) for compatibility.
        ref_pixels = 1024.0 * 1024.0
        img_pixels = max(1.0, float(width) * float(height))
        pixel_ratio = img_pixels / ref_pixels
        img_mp = img_pixels / 1_000_000.0

        # Calibrated resolution range:
        # ~0.30MP -> denoise_min
        # ~3.70MP -> denoise_max (at adaptive_level=5).
        low_mp = 0.30
        high_mp = 3.70
        mp_span = max(1e-8, high_mp - low_mp)
        normalized = self._clamp((img_mp - low_mp) / mp_span, 0.0, 1.0)

        min_d = float(denoise_min)
        max_d = float(denoise_max)
        if max_d < min_d:
            min_d, max_d = max_d, min_d

        # Resolution-derived base denoise (level-neutral baseline).
        denoise_range = max_d - min_d
        base_denoise = min_d + denoise_range * normalized

        # Stronger level spread with level 5 as neutral:
        # level 10 -> about +0.02 (for default range 0.08..0.15)
        # level 1  -> about -0.018
        level = int(adaptive_level)
        up_spread = denoise_range * 0.285714
        down_spread = denoise_range * 0.257143
        if level >= 5:
            t = (float(level) - 5.0) / 5.0
            level_offset = t * up_spread
        else:
            t = (5.0 - float(level)) / 4.0
            level_offset = -(t * down_spread)

        final_denoise = self._clamp(base_denoise + level_offset, 0.0001, 1.0)

        report = (
            f"width={width} height={height} img_pixels={int(img_pixels)} "
            f"img_mp={img_mp:.6f} auto_ref={int(ref_pixels)} ratio={pixel_ratio:.6f} "
            f"adaptive_level={int(adaptive_level)} normalized={normalized:.6f} "
            f"base_denoise={base_denoise:.4f} level_offset={level_offset:.4f} "
            f"denoise={final_denoise:.4f} min={min_d:.4f} max={max_d:.4f}"
        )

        return float(final_denoise), width, height, float(pixel_ratio), report


NODE_CLASS_MAPPINGS = {
    "Synthid-Bypass-Facedetailer": SEGSDetailerModelSwap,
    "SEGSDetailerModelSwap": SEGSDetailerModelSwap,
    "Synthid-Bypass-AdaptiveDenoise": SynthidBypassAdaptiveDenoise,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Synthid-Bypass-Facedetailer": "Synthid-Bypass-Facedetailer",
    "SEGSDetailerModelSwap": "Synthid-Bypass-Facedetailer (Legacy)",
    "Synthid-Bypass-AdaptiveDenoise": "Synthid-Bypass-AdaptiveDenoise",
}
