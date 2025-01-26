import pytest
import tempfile
from pathlib import Path
from contextforge import compile_project, ContextForgeHandler

@pytest.fixture
def temp_project_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

@pytest.fixture
def sample_files(temp_project_dir):
    # Create test files
    (temp_project_dir / "file1.txt").write_text("Sample content 1")
    (temp_project_dir / "file2.md").write_text("# Sample markdown")
    (temp_project_dir / "large_file.txt").write_text("x" * 1024 * 1024)  # 1MB file
    (temp_project_dir / "ignored.xyz").write_text("Should be ignored")
    return temp_project_dir

def test_basic_compilation(temp_project_dir, sample_files):
    output_dir = temp_project_dir / "output"
    result = compile_project(
        project_path=str(sample_files),
        output_path=str(output_dir),
    )
    
    assert result.success
    assert (output_dir / "file1.txt").exists()
    assert (output_dir / "file2.md").exists()

def test_file_size_limits(temp_project_dir, sample_files):
    output_dir = temp_project_dir / "output"
    handler = ContextForgeHandler(
        max_file_size_mb=0.5  # Set max file size to 0.5MB
    )
    
    result = compile_project(
        project_path=str(sample_files),
        output_path=str(output_dir),
        handler=handler
    )
    
    assert result.success
    assert not (output_dir / "large_file.txt").exists()
    assert (output_dir / "file1.txt").exists()

def test_extension_filtering(temp_project_dir, sample_files):
    output_dir = temp_project_dir / "output"
    handler = ContextForgeHandler(
        included_extensions=[".txt"]
    )
    
    result = compile_project(
        project_path=str(sample_files),
        output_path=str(output_dir),
        handler=handler
    )
    
    assert result.success
    assert (output_dir / "file1.txt").exists()
    assert not (output_dir / "file2.md").exists()
    assert not (output_dir / "ignored.xyz").exists()

def test_custom_output_format(temp_project_dir, sample_files):
    output_dir = temp_project_dir / "output"
    handler = ContextForgeHandler(
        output_format="json"
    )
    
    result = compile_project(
        project_path=str(sample_files),
        output_path=str(output_dir),
        handler=handler
    )
    
    assert result.success
    # Check if output is in JSON format
    output_file = output_dir / "context.json"
    assert output_file.exists()
    assert output_file.read_text().startswith("{")
    assert output_file.read_text().endswith("}")

