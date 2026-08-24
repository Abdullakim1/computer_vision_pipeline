"""CineForge CLI Tests"""

import subprocess
import sys
from pathlib import Path


def test_info_command():
    """Test that info command works."""
    result = subprocess.run(
        [sys.executable, "-m", "src.main", "info"],
        cwd=Path("/home/kim/computer_vision_pipeline"),
        capture_output=True,
        text=True,
    )
    print("INFO command output:")
    print(result.stdout)
    print(result.stderr)
    assert result.returncode == 0
    print("✓ Info command works")


def test_stats_command():
    """Test that stats command works."""
    result = subprocess.run(
        [sys.executable, "-m", "src.main", "stats"],
        cwd=Path("/home/kim/computer_vision_pipeline"),
        capture_output=True,
        text=True,
    )
    print("STATS command output:")
    print(result.stdout)
    print(result.stderr)
    assert result.returncode == 0
    print("✓ Stats command works")


def test_demo_command():
    """Test that demo command works."""
    result = subprocess.run(
        [sys.executable, "-m", "src.main", "demo", "--outdir", "/tmp/demo_output"],
        cwd=Path("/home/kim/computer_vision_pipeline"),
        capture_output=True,
        text=True,
        timeout=30,
    )
    print("DEMO command output:")
    print(result.stdout)
    print(result.stderr)
    assert result.returncode == 0
    print("✓ Demo command works")


def test_showcase_command():
    """Test that showcase command works."""
    result = subprocess.run(
        [sys.executable, "-m", "src.main", "showcase", "--outdir", "/tmp/showcase_output"],
        cwd=Path("/home/kim/computer_vision_pipeline"),
        capture_output=True,
        text=True,
        timeout=600,
    )
    print("SHOWCASE command output:")
    print(result.stdout)
    print(result.stderr)
    assert result.returncode == 0
    print("✓ Showcase command works")


if __name__ == "__main__":
    print("Running CineForge Tests...")
    print("=" * 50)
    
    test_info_command()
    test_stats_command()
    test_demo_command()
    test_showcase_command()
    
    print("=" * 50)
    print("All tests passed! ✓")
