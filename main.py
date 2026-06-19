import argparse

from config.loader import load_config, list_profiles, PipelineConfig
from data.dataset_loader import get_child_psrb_samples
from data.preprocessor import DataPipeline
from evaluation.metrics import calculate_metrics
from evaluation.normalization import PersianNormalizer
from evaluation.postprocessing.llm_corrector import LLMCorrector
from evaluation.postprocessing.rule_based import RuleBasedCorrector
from models.model_factory import ModelFactory

def _build_model(cfg: PipelineConfig):
    model = ModelFactory.from_config(cfg.model)

    if cfg.data.use_vocab_boosting and cfg.model.type == "vosk":
        if cfg.data.source == "cached":
            vocab: set[str] = set()
            for _, _, transcript in get_child_psrb_samples(cache_dir=cfg.data.cache_dir):
                words = transcript.replace(".", "").replace(",", "").replace("?", "").split()
                vocab.update(words)
            model.set_vocabulary(list(vocab))
        else:
            print("Warning: Vocab boosting is only supported for the 'cached' data source.")

    model.load()
    return model


def _build_corrector(cfg: PipelineConfig):
    post = cfg.postprocessing

    if post.type == "classic":
        return RuleBasedCorrector(post.lexicon_path)

    if post.type == "llm":
        return LLMCorrector(
            model_name=post.llm_model,
            prompt_path=post.prompt_path,
            lexicon_path=post.lexicon_path,
            rate_limit_delay=post.rate_limit_delay,
        )

    return None


def _get_samples(cfg: PipelineConfig, pipeline: DataPipeline, split: str = "all"):
    source = cfg.data.source

    if source == "hf_stream":
        yield from pipeline.stream_from_hf(dataset_name="PartAI/PSRB", filter_child=True)
    elif source == "cached":
        yield from pipeline.from_cached_dataset(cache_dir=cfg.data.cache_dir, split=split)
    elif source == "mic":
        yield pipeline.from_mic(duration_seconds=5)
    else:
        # Treat source as a local file path
        yield pipeline.from_local_file(source)


def _print_sample_result(
    index: int,
    ref_transcript: str,
    hypothesis: str,
    final_hyp: str,
    metrics: dict,
    post_type: str,
) -> None:
    print(f"Sample {index}:")
    print(f"  Ref: {ref_transcript}", flush=True)

    if post_type == "none":
        print(f"  Hyp: {hypothesis}", flush=True)
    else:
        print(f"  Raw: {hypothesis}", flush=True)
        print(f"  Corrected: {final_hyp}", flush=True)

    print(f"  WER: {metrics['wer']:.2%} | CER: {metrics['cer']:.2%} | Lev: {metrics['levenshtein']}", flush=True)


def run_pipeline(cfg: PipelineConfig, split: str = "all") -> None:
    print(f"Running profile: {cfg.profile_name}")

    model = _build_model(cfg)

    pipeline = DataPipeline(
        audio_cfg=cfg.model.audio,
        apply_noise_reduction=cfg.data.use_noise_reduction,
    )

    normalizer = PersianNormalizer()
    corrector = _build_corrector(cfg)

    print("Starting ASR...")

    total_wer = 0.0
    total_cer = 0.0
    total_lev = 0.0
    num_samples = 0

    samples_run = []
    import re
    for i, (audio_array, ref_transcript) in enumerate(_get_samples(cfg, pipeline, split)):
        sample_idx = i + 1
        if sample_idx in cfg.run.exclude_samples:
            continue
        # Check if reference contains English characters
        if re.search(r'[a-zA-Z]', ref_transcript):
            continue
        samples_run.append((sample_idx, audio_array, ref_transcript))

    for idx, audio_array, ref_transcript in samples_run:
        if num_samples >= cfg.run.max_samples:
            break

        print(f"Sample {idx}: transcribing...", end="", flush=True)
        hypothesis = model.transcribe(audio_array, language=cfg.model.language)

        final_hyp = corrector.correct(hypothesis) if corrector else hypothesis

        # Check if hypothesis contains English characters (corrupted/translated text)
        if re.search(r'[a-zA-Z]', final_hyp):
            print("\r", end="")
            continue

        norm_ref = normalizer.normalize(ref_transcript)
        norm_hyp = normalizer.normalize(final_hyp)
        metrics = calculate_metrics(norm_ref, norm_hyp)

        print("\r", end="")  # clear the "transcribing..." line
        _print_sample_result(
            idx, ref_transcript, hypothesis, final_hyp, metrics, cfg.postprocessing.type
        )
        
        total_wer += metrics['wer']
        total_cer += metrics['cer']
        total_lev += metrics['levenshtein']
        num_samples += 1

    if num_samples > 0:
        avg_wer = total_wer / num_samples
        avg_cer = total_cer / num_samples
        avg_lev = total_lev / num_samples
        
        import json
        import os
        results_file = "evaluation_results.json"
        results = {}
        if os.path.exists(results_file):
            with open(results_file, "r") as f:
                try:
                    results = json.load(f)
                except json.JSONDecodeError:
                    pass
                    
        profile_key = cfg.profile_name
        if split != "all":
            profile_key = f"{profile_key} ({split})"
            
        results[profile_key] = {
            "model": cfg.model.type,
            "avg_wer": avg_wer,
            "avg_cer": avg_cer,
            "avg_lev": avg_lev,
            "samples": num_samples
        }
        
        with open(results_file, "w") as f:
            json.dump(results, f, indent=4)
        
        print(f"Saved results to {results_file}. Avg WER: {avg_wer:.2%}, Avg CER: {avg_cer:.2%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="BSC Thesis ASR — Profile-Based Runner",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--profile", type=str, default=None,
        help="Config profile to run (see config/profiles.yaml).",
    )
    parser.add_argument(
        "--list-profiles", action="store_true",
        help="Print available profile names and exit.",
    )

    parser.add_argument(
        "--models-config", type=str, default="config/models.yaml",
        help="Path to models.yaml.",
    )
    parser.add_argument(
        "--profiles-config", type=str, default="config/profiles.yaml",
        help="Path to profiles.yaml.",
    )

    parser.add_argument("--model", type=str, help="Override model type.")
    parser.add_argument("--source", type=str, help="Override data source.")
    parser.add_argument("--language", type=str, help="Override language.")
    parser.add_argument("--split", type=str, default="all", choices=["all", "train", "test"], help="Dataset split to evaluate on.")
    parser.add_argument("--max-samples", type=int, dest="max_samples", help="Override max samples.")
    parser.add_argument("--exclude-samples", type=str, help="Comma-separated 1-based indices of samples to exclude.")

    args = parser.parse_args()

    if args.list_profiles:
        profiles = list_profiles(args.profiles_config)
        print("Available profiles:")
        for p in profiles:
            print(f"  - {p}")
        raise SystemExit(0)

    try:
        cfg = load_config(
            profile_name=args.profile,
            models_path=args.models_config,
            profiles_path=args.profiles_config,
            cli_overrides=args,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        raise SystemExit(1)

    print(f"Loading configuration profile: '{cfg.profile_name}'")
    run_pipeline(cfg, split=args.split)
