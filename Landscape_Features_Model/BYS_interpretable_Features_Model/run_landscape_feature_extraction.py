from extract_landscape_depth_features import main as run_depth
from extract_landscape_handcrafted_features import main as run_handcrafted
from extract_landscape_segformer_features import main as run_segformer
from merge_landscape_interpretable_features import main as run_merge


def main() -> None:
    print("Step 1/4: Extract SegFormer semantic features")
    run_segformer()

    print("\nStep 2/4: Extract Depth Anything V2 features")
    run_depth()

    print("\nStep 3/4: Extract handcrafted color/composition features")
    run_handcrafted()

    print("\nStep 4/4: Merge all feature tables")
    run_merge()


if __name__ == "__main__":
    main()
