#!/usr/bin/env python3
"""
Bentham Dataset to IAM Format Converter
=======================================

Converts Bentham dataset structure to IAM format:
- Line images from Images/Lines/ 
- Text files from Transcriptions/
- Creates single gt.txt file in IAM format

Input structure:
BenthamDatasetR0-GT/
├───Images
│   ├───Lines      (line images)
│   └───Pages      (empty)
├───PAGE           (XML files)
├───Partitions
└───Transcriptions (TXT files)

Output structure:
bentham_iam_format/
├───images/        (all line images)
└───gt.txt         (single ground truth file)
"""

import os
import shutil
from pathlib import Path
import argparse

def match_images_with_transcriptions(images_dir, transcriptions_dir):
    """
    Match line images with their corresponding transcription files

    Args:
        images_dir: Path to Images/Lines directory
        transcriptions_dir: Path to Transcriptions directory

    Returns:
        List of (image_path, text_content) tuples
    """

    # Get all image files
    image_extensions = ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']
    image_files = []

    for ext in image_extensions:
        image_files.extend(list(Path(images_dir).glob(f'*{ext}')))
        image_files.extend(list(Path(images_dir).glob(f'*{ext.upper()}')))

    print(f"Found {len(image_files)} image files in {images_dir}")

    # Get all text files  
    text_files = list(Path(transcriptions_dir).glob('*.txt'))
    print(f"Found {len(text_files)} text files in {transcriptions_dir}")

    # Match images with text files
    matched_pairs = []
    unmatched_images = []

    for image_file in image_files:
        # Try different matching strategies
        corresponding_text = None
        image_stem = image_file.stem

        # Strategy 1: Exact stem match
        for text_file in text_files:
            if text_file.stem == image_stem:
                corresponding_text = text_file
                break

        # Strategy 2: Remove common prefixes/suffixes and match
        if not corresponding_text:
            # Clean image stem (remove common patterns)
            clean_image_stem = image_stem.replace('-line', '').replace('_line', '').replace('.line', '')

            for text_file in text_files:
                clean_text_stem = text_file.stem.replace('-line', '').replace('_line', '').replace('.line', '')
                if clean_image_stem == clean_text_stem:
                    corresponding_text = text_file
                    break

        # Strategy 3: Partial matching (if image name contains text name or vice versa)
        if not corresponding_text:
            for text_file in text_files:
                if (image_stem in text_file.stem or text_file.stem in image_stem or
                    image_stem.replace('-', '_') == text_file.stem.replace('-', '_')):
                    corresponding_text = text_file
                    break

        if corresponding_text:
            # Read text content
            try:
                with open(corresponding_text, 'r', encoding='utf-8') as f:
                    text_content = f.read().strip()

                matched_pairs.append((image_file, text_content))
                print(f"  ✓ Matched: {image_file.name} → {corresponding_text.name}")

            except Exception as e:
                print(f"  ✗ Error reading {corresponding_text}: {e}")
        else:
            unmatched_images.append(image_file)
            print(f"  ✗ No match found for: {image_file.name}")

    print(f"\nMatching summary:")
    print(f"  Matched pairs: {len(matched_pairs)}")
    print(f"  Unmatched images: {len(unmatched_images)}")

    if unmatched_images:
        print(f"\nUnmatched images:")
        for img in unmatched_images[:10]:  # Show first 10
            print(f"    {img.name}")
        if len(unmatched_images) > 10:
            print(f"    ... and {len(unmatched_images) - 10} more")

    return matched_pairs

def create_iam_format_dataset(bentham_root, output_dir):
    """
    Convert Bentham dataset to IAM format

    Args:
        bentham_root: Path to BenthamDatasetR0-GT directory
        output_dir: Path to output directory for IAM format
    """

    bentham_path = Path(bentham_root)
    output_path = Path(output_dir)

    # Check input directories
    images_lines_dir = bentham_path / 'Images' / 'Lines'
    transcriptions_dir = bentham_path / 'Transcriptions'

    if not images_lines_dir.exists():
        print(f"Error: {images_lines_dir} not found")
        return False

    if not transcriptions_dir.exists():
        print(f"Error: {transcriptions_dir} not found")
        return False

    print(f"Converting Bentham dataset to IAM format...")
    print(f"Input: {bentham_path}")
    print(f"Output: {output_path}")

    # Create output directories
    output_images_dir = output_path / 'images'
    output_images_dir.mkdir(parents=True, exist_ok=True)

    # Match images with transcriptions
    matched_pairs = match_images_with_transcriptions(images_lines_dir, transcriptions_dir)

    if not matched_pairs:
        print("No matched pairs found! Check your file naming patterns.")
        return False

    # Create gt.txt file and copy images
    gt_file = output_path / 'gt.txt'

    print(f"\nCreating IAM format dataset...")

    with open(gt_file, 'w', encoding='utf-8') as f:
        for i, (image_file, text_content) in enumerate(matched_pairs):
            # Create standardized filename
            new_image_name = f"bentham_{i:06d}{image_file.suffix}"

            # Copy image to output directory
            shutil.copy2(image_file, output_images_dir / new_image_name)

            # Write entry to gt.txt (IAM format: filename<space>text)
            # Remove file extension for the gt.txt entry (IAM convention)
            image_id = f"bentham_{i:06d}"
            f.write(f"{image_id} {text_content}\n")

            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1} images...")

    print(f"\n Conversion completed!")
    print(f"Created IAM format dataset in: {output_path}")
    print(f"  - {len(matched_pairs)} images in images/")
    print(f"  - Ground truth file: gt.txt")

    # Show sample entries
    print(f"\nSample gt.txt entries:")
    with open(gt_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i < 5:  # Show first 5 entries
                parts = line.strip().split(' ', 1)
                if len(parts) == 2:
                    print(f"  {parts[0]} → {parts[1][:60]}...")
            else:
                break

    return True

def create_train_val_split(iam_dataset_dir, train_ratio=0.8):
    """
    Create train/validation split for the IAM format dataset

    Args:
        iam_dataset_dir: Directory containing images/ and gt.txt
        train_ratio: Ratio of data to use for training (default: 0.8)
    """

    dataset_path = Path(iam_dataset_dir)
    gt_file = dataset_path / 'gt.txt'

    if not gt_file.exists():
        print(f"Error: {gt_file} not found")
        return

    # Read all entries
    with open(gt_file, 'r', encoding='utf-8') as f:
        all_entries = f.readlines()

    # Shuffle and split
    import random
    random.seed(42)  # For reproducible splits
    random.shuffle(all_entries)

    split_point = int(len(all_entries) * train_ratio)
    train_entries = all_entries[:split_point]
    val_entries = all_entries[split_point:]

    # Write train and validation files
    train_file = dataset_path / 'gt_train.txt'
    val_file = dataset_path / 'gt_val.txt'

    with open(train_file, 'w', encoding='utf-8') as f:
        f.writelines(train_entries)

    with open(val_file, 'w', encoding='utf-8') as f:
        f.writelines(val_entries)

    print(f"\n Created train/validation split:")
    print(f"  Train: {len(train_entries)} samples → gt_train.txt")
    print(f"  Validation: {len(val_entries)} samples → gt_val.txt")

def main():
    parser = argparse.ArgumentParser(description='Convert Bentham dataset to IAM format')
    parser.add_argument('--bentham_root', required=True, help='Path to BenthamDatasetR0-GT directory')
    parser.add_argument('--output_dir', required=True, help='Output directory for IAM format dataset')
    parser.add_argument('--create_split', action='store_true', help='Create train/validation split')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Training data ratio (default: 0.8)')

    args = parser.parse_args()

    # Convert to IAM format
    success = create_iam_format_dataset(args.bentham_root, args.output_dir)

    if success and args.create_split:
        create_train_val_split(args.output_dir, args.train_ratio)

if __name__ == "__main__":
    # Example usage if run without arguments
    import sys

    if len(sys.argv) == 1:
        print("Bentham to IAM Format Converter")
        print("=" * 40)
        print()
        print("Usage examples:")
        print("1. Basic conversion:")
        print("   python bentham_to_iam.py --bentham_root /path/to/BenthamDatasetR0-GT --output_dir ./bentham_iam")
        print()
        print("2. With train/val split:")
        print("   python bentham_to_iam.py --bentham_root /path/to/BenthamDatasetR0-GT --output_dir ./bentham_iam --create_split")
        print()
        print("Interactive mode:")
        bentham_root = input("Enter path to BenthamDatasetR0-GT directory: ").strip()
        output_dir = input("Enter output directory path: ").strip()
        create_split = input("Create train/validation split? (y/n): ").strip().lower() == 'y'

        if bentham_root and output_dir:
            success = create_iam_format_dataset(bentham_root, output_dir)
            if success and create_split:
                create_train_val_split(output_dir)
    else:
        main()
