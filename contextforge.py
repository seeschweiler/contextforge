import os
import argparse
from tqdm import tqdm
import markdown2
import fnmatch
import datetime
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

def get_file_content(file_path):
    """Read and return the content of a file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def is_binary(file_path):
    """Check if a file is binary."""
    try:
        with open(file_path, 'tr') as check_file:
            check_file.read()
            return False
    except:
        return True

def get_language(file_extension):
    """Map file extension to Markdown code block language."""
    extension_map = {
        # ... (keep the existing extension_map here)
    }
    return extension_map.get(file_extension.lower(), '')

def load_cfignore(project_path):
    """Load and parse the .cfignore file."""
    cfignore_path = os.path.join(project_path, '.cfignore')
    ignore_patterns = []
    if os.path.exists(cfignore_path):
        with open(cfignore_path, 'r') as f:
            ignore_patterns = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return ignore_patterns

def should_ignore(file_path, ignore_patterns):
    """Check if a file should be ignored based on .cfignore patterns."""
    for pattern in ignore_patterns:
        if fnmatch.fnmatch(file_path, pattern):
            return True
    return False

def process_file(file_info):
    """Process a single file and return its content."""
    file_path, relative_path, ignore_patterns, max_file_size = file_info
    
    if should_ignore(relative_path, ignore_patterns):
        return None
    
    if os.path.getsize(file_path) > max_file_size:
        return f"## File: {relative_path}\n\nFile exceeds size limit. Content not included.\n\n"
    
    content = f"## File: {relative_path}\n\nLocation: `{file_path}`\n\n"
    
    if is_binary(file_path):
        content += "```\nBinary file, content not displayed.\n```\n\n"
    else:
        file_content = get_file_content(file_path)
        file_extension = os.path.splitext(file_path)[1]
        
        if file_extension.lower() in ['.md', '.markdown']:
            html_content = markdown2.markdown(file_content)
            content += html_content + "\n\n"
        else:
            language = get_language(file_extension)
            content += f"```{language}\n{file_content}\n```\n\n"
    
    return content

def compile_project(project_path, output_file, output_format='markdown', max_file_size=1000000):
    """Compile project files into a single file."""
    ignore_patterns = load_cfignore(project_path)
    
    start_time = datetime.datetime.now()
    total_files = 0
    processed_files = 0
    ignored_files = 0
    
    file_contents = []
    
    with ThreadPoolExecutor() as executor:
        future_to_file = {}
        for root, _, files in os.walk(project_path):
            for file in files:
                total_files += 1
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, project_path)
                future = executor.submit(process_file, (file_path, relative_path, ignore_patterns, max_file_size))
                future_to_file[future] = relative_path
        
        for future in tqdm(as_completed(future_to_file), total=len(future_to_file), desc="Forging context", unit="file"):
            content = future.result()
            if content:
                file_contents.append(content)
                processed_files += 1
            else:
                ignored_files += 1
    
    end_time = datetime.datetime.now()
    compilation_time = (end_time - start_time).total_seconds()
    
    metadata = {
        "project_name": os.path.basename(project_path),
        "compilation_date": end_time.isoformat(),
        "total_files": total_files,
        "processed_files": processed_files,
        "ignored_files": ignored_files,
        "compilation_time": compilation_time
    }
    
    with open(output_file, 'w', encoding='utf-8') as out_file:
        if output_format == 'markdown':
            out_file.write(f"# ContextForge Compilation: {metadata['project_name']}\n\n")
            out_file.write("## Compilation Metadata\n\n")
            out_file.write(f"- Compilation Date: {metadata['compilation_date']}\n")
            out_file.write(f"- Total Files: {metadata['total_files']}\n")
            out_file.write(f"- Processed Files: {metadata['processed_files']}\n")
            out_file.write(f"- Ignored Files: {metadata['ignored_files']}\n")
            out_file.write(f"- Compilation Time: {metadata['compilation_time']:.2f} seconds\n\n")
            out_file.write("## File Contents\n\n")
            for content in file_contents:
                out_file.write(content)
        elif output_format == 'html':
            out_file.write("<html><body>")
            out_file.write(f"<h1>ContextForge Compilation: {metadata['project_name']}</h1>")
            out_file.write("<h2>Compilation Metadata</h2>")
            out_file.write("<ul>")
            for key, value in metadata.items():
                out_file.write(f"<li>{key}: {value}</li>")
            out_file.write("</ul>")
            out_file.write("<h2>File Contents</h2>")
            for content in file_contents:
                out_file.write(markdown2.markdown(content))
            out_file.write("</body></html>")
        elif output_format == 'json':
            json.dump({"metadata": metadata, "contents": file_contents}, out_file, indent=2)

def main():
    parser = argparse.ArgumentParser(
        description="ContextForge: Compile project files into a single AI-ready file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,  # This allows us to format the epilog
        epilog="""
Examples:
  Compile current directory to default output file:
    %(prog)s

  Compile specific project to custom output file:
    %(prog)s /path/to/project custom_output.md

  Compile to HTML format:
    %(prog)s -f html

  Compile with 500KB max file size:
    %(prog)s -m 500000

  Compile to JSON format with 2MB max file size:
    %(prog)s -f json -m 2000000 /path/to/project output.json

  Get help:
    %(prog)s -h
        """
    )
    parser.add_argument("project_path", nargs='?', default=".", help="Path to the project folder (default: current directory)")
    parser.add_argument("output_file", nargs='?', default="context_forge_output.md", help="Path to the output file")
    parser.add_argument("-f", "--format", choices=['markdown', 'html', 'json'], default='markdown', help="Output format (default: %(default)s)")
    parser.add_argument("-m", "--max-file-size", type=int, default=1000000, help="Maximum file size in bytes to include in compilation (default: %(default)s)")
    args = parser.parse_args()

    compile_project(args.project_path, args.output_file, args.format, args.max_file_size)
    print(f"ContextForge complete. AI-ready context saved to {args.output_file}")

if __name__ == "__main__":
    main()