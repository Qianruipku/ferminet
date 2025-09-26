import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description='Plot learning curve from training statistics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python learning_curve.py                              # Use default settings (JPG format)
  python learning_curve.py --ylim -1.5 -1.0           # Set Y-axis limits
  python learning_curve.py --xlim 0 1000              # Set X-axis limits
  python learning_curve.py --ylim -2.0 -0.5 --xlim 100 500  # Set both axis limits
  python learning_curve.py -f my_stats.csv --ylim -2.0 -0.5  # Custom file and Y limits
  python learning_curve.py --format pdf -o curve.pdf   # Save as PDF vector format
  python learning_curve.py --format svg --figsize 10 6 # SVG with custom size
        ''')
    parser.add_argument('--file', '-f', type=str, default="train_stats.csv",
                       help='Path to the training statistics CSV file (default: train_stats.csv)')
    parser.add_argument('--ylim', nargs=2, type=float, default=None, metavar=('YMIN', 'YMAX'),
                       help='Set Y-axis limits: --ylim YMIN YMAX (e.g., --ylim -1.5 -1.0)')
    parser.add_argument('--xlim', nargs=2, type=float, default=None, metavar=('XMIN', 'XMAX'),
                       help='Set X-axis limits: --xlim XMIN XMAX (e.g., --xlim 0 1000)')
    parser.add_argument('--output', '-o', type=str, default="learning_curve.jpg",
                       help='Output image file path (default: learning_curve.jpg)')
    parser.add_argument('--format', type=str, choices=['pdf', 'svg', 'eps', 'png', 'jpg'], default='jpg',
                       help='Output format: pdf, svg, eps, png, or jpg (default: jpg)')
    parser.add_argument('--dpi', type=int, default=300,
                       help='Image resolution in DPI for raster formats (default: 300)')
    parser.add_argument('--figsize', nargs=2, type=float, default=[12, 8], metavar=('WIDTH', 'HEIGHT'),
                       help='Figure size: --figsize WIDTH HEIGHT (default: 12 8)')
    return parser.parse_args()

args = parse_args()
file_path = args.file
df = pd.read_csv(file_path)

x_data = df.iloc[:, 0]
y_data = df.iloc[:, 2]


plt.figure(figsize=tuple(args.figsize))
plt.plot(x_data, y_data, linewidth=1.5, color='blue', alpha=0.8)

# Set Y-axis limits if specified
if args.ylim is not None:
    plt.ylim(args.ylim[0], args.ylim[1])
    print(f"Y-axis limits set to: {args.ylim[0]} - {args.ylim[1]}")

# Set X-axis limits if specified
if args.xlim is not None:
    plt.xlim(args.xlim[0], args.xlim[1])
    print(f"X-axis limits set to: {args.xlim[0]} - {args.xlim[1]}")

plt.xlabel('Step', fontsize=14)
plt.ylabel('EWMean', fontsize=14)
plt.title('Training Statistics: Step vs EWMean', fontsize=16)

plt.grid(True, alpha=0.3)

plt.tight_layout()

output_path = args.output
# Automatically determine format from file extension if not explicitly set
if args.format == 'jpg' and not output_path.endswith('.jpg'):
    if '.' not in output_path:
        output_path = f"{output_path}.jpg"
    else:
        # Replace extension with jpg
        output_path = '.'.join(output_path.split('.')[:-1]) + '.jpg'

# Save with appropriate settings for vector vs raster formats
if args.format in ['pdf', 'svg', 'eps']:
    # Vector formats - DPI not needed
    plt.savefig(output_path, format=args.format, bbox_inches='tight')
    print(f"Vector format ({args.format.upper()}) saved - scalable without quality loss")
else:
    # Raster formats - use DPI
    plt.savefig(output_path, format=args.format, dpi=args.dpi, bbox_inches='tight')
    print(f"Raster format ({args.format.upper()}) saved at {args.dpi} DPI")

print(f"Input file: {file_path}")
print(f"Step range: {x_data.min()} - {x_data.max()}")
print(f"EWMean range: {y_data.min():.6f} - {y_data.max():.6f}")
print(f"Image saved to: {output_path}")

plt.close()