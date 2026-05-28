from vision_lens.pipeline import run_vit_attention


def main() -> None:
    result = run_vit_attention("configs/vit_attention.example.yaml")
    print(f"saved {len(result.output_paths)} files to {result.config.output.directory}")


if __name__ == "__main__":
    main()
