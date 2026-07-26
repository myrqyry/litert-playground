import argparse
import os
import tarfile
import tempfile
import tensorflow as tf


def extract_mlxfn(mlxfn_path: str, output_dir: str):
    """Extract saved model from .mlxfn bundle (tar archive)."""
    with tarfile.open(mlxfn_path, 'r') as tar:
        tar.extractall(path=output_dir)


def convert_to_tflite(saved_model_dir: str, output_path: str):
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    tflite_model = converter.convert()
    with open(output_path, 'wb') as f:
        f.write(tflite_model)


def main():
    parser = argparse.ArgumentParser(description='Convert .mlxfn to .tflite')
    parser.add_argument('input', help='Path to .mlxfn file')
    parser.add_argument('-o', '--output', default='magenta_realtime_2.tflite',
                        help='Output .tflite path')
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmpdir:
        saved_model_dir = os.path.join(tmpdir, 'saved_model')
        print(f'Extracting {args.input}...')
        extract_mlxfn(args.input, saved_model_dir)
        print(f'Converting to {args.output}...')
        convert_to_tflite(saved_model_dir, args.output)
        print(f'Done: {args.output}')


if __name__ == '__main__':
    main()
