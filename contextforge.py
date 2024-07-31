import os
import argparse
from tqdm import tqdm
import markdown2
import fnmatch
import datetime
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import tiktoken
import xml.etree.ElementTree as ET
from xml.dom import minidom
from pathlib import Path
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class ContextForgeHandler(FileSystemEventHandler):
    def __init__(self, project_path, output_file, output_format, max_file_size, allowed_extensions):
        self.project_path = project_path
        self.output_file = Path(output_file).resolve()
        self.output_format = output_format
        self.max_file_size = max_file_size
        self.allowed_extensions = allowed_extensions
        self.last_compile_time = 0
        self.cooldown = 5  # 5 seconds cooldown

    def on_any_event(self, event):
        if event.is_directory:
            return

        # Ignore changes to the output file itself
        if Path(event.src_path).resolve() == self.output_file:
            return

        current_time = time.time()
        if current_time - self.last_compile_time > self.cooldown:
            if event.event_type in ['created', 'modified', 'deleted']:
                print(f"Detected change in {event.src_path}. Recompiling...")
                compile_project(self.project_path, str(self.output_file), self.output_format, 
                                self.max_file_size, self.allowed_extensions)
                self.last_compile_time = current_time

def watch_project(project_path, output_file, output_format='markdown', max_file_size=1000000, allowed_extensions=None):
    event_handler = ContextForgeHandler(project_path, output_file, output_format, max_file_size, allowed_extensions)
    observer = Observer()
    observer.schedule(event_handler, project_path, recursive=True)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

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
        # Common languages
        '.py': 'python',
        '.js': 'javascript',
        '.jsx': 'jsx',
        '.ts': 'typescript',
        '.tsx': 'tsx',
        '.html': 'html',
        '.css': 'css',
        '.scss': 'scss',
        '.sass': 'sass',
        '.less': 'less',
        '.java': 'java',
        '.c': 'c',
        '.cpp': 'cpp',
        '.cs': 'csharp',
        '.go': 'go',
        '.rb': 'ruby',
        '.php': 'php',
        '.swift': 'swift',
        '.kt': 'kotlin',
        '.rs': 'rust',
        '.scala': 'scala',
        
        # Shell and scripting
        '.sh': 'bash',
        '.bash': 'bash',
        '.zsh': 'zsh',
        '.fish': 'fish',
        '.ps1': 'powershell',
        '.bat': 'batch',
        '.cmd': 'batch',
        
        # Markup and data
        '.xml': 'xml',
        '.json': 'json',
        '.yml': 'yaml',
        '.yaml': 'yaml',
        '.toml': 'toml',
        '.md': 'markdown',
        '.tex': 'latex',
        
        # Database
        '.sql': 'sql',
        '.plsql': 'plsql',
        
        # Web technologies
        '.vue': 'vue',
        '.svelte': 'svelte',
        '.graphql': 'graphql',
        
        # Functional languages
        '.hs': 'haskell',
        '.elm': 'elm',
        '.erl': 'erlang',
        '.ex': 'elixir',
        '.exs': 'elixir',
        '.clj': 'clojure',
        '.lisp': 'lisp',
        '.scm': 'scheme',
        
        # Other languages
        '.lua': 'lua',
        '.pl': 'perl',
        '.r': 'r',
        '.dart': 'dart',
        '.f': 'fortran',
        '.f90': 'fortran',
        '.jl': 'julia',
        '.m': 'matlab',
        '.mm': 'objectivec',
        '.vb': 'vbnet',
        '.groovy': 'groovy',
        '.tcl': 'tcl',
        '.asm': 'assembly',
        '.pas': 'pascal',
        '.d': 'd',
        '.nim': 'nim',
        '.zig': 'zig',
        '.v': 'v',
        '.ada': 'ada',
        '.fs': 'fsharp',
        '.cob': 'cobol',
        '.coffee': 'coffeescript',
        
        # Configuration files
        '.ini': 'ini',
        '.cfg': 'ini',
        '.conf': 'ini',
        '.properties': 'properties',
        
        # Build and package management
        '.gradle': 'gradle',
        '.maven': 'maven',
        '.ant': 'ant',
        '.cmake': 'cmake',
        '.dockerfile': 'dockerfile',
        '.makefile': 'makefile',
        '.mk': 'makefile',
        
        # Version control
        '.gitignore': 'gitignore',
        '.gitattributes': 'gitattributes',
        
        # Misc
        '.log': 'log',
        '.csv': 'csv'
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

def should_ignore(path, ignore_patterns, output_file, allowed_extensions):
    """Check if a file or directory should be ignored based on .cfignore patterns, output file, and allowed extensions."""
    # Normalize paths for consistent comparison
    path = Path(path).resolve()
    if output_file:
        output_file = Path(output_file).resolve()

    # Check if the path is the output file
    if output_file:
        if path.exists() and output_file.exists() and path.samefile(output_file):
            return True
        # If the output file doesn't exist yet, compare the normalized paths
        if path == output_file:
            return True
    
    # Get the relative path from the project root
    rel_path = path.relative_to(Path.cwd())
    
    for pattern in ignore_patterns:
        # Remove leading and trailing whitespace and slashes
        pattern = pattern.strip().strip('/')
        
        # Convert pattern to Path object for easier manipulation
        pattern_path = Path(pattern)
        
        # Check if the pattern is meant to match from the root
        is_root_pattern = pattern.startswith('/')
        
        # Create different variations of the path to check against
        paths_to_check = [
            str(rel_path),
            str(path),
            str(rel_path).lstrip('./'),
            path.name,
            *(str(rel_path.relative_to(part)) for part in rel_path.parents if part != Path('.'))
        ]
        
        for check_path in paths_to_check:
            # For root patterns, only check against the full relative path
            if is_root_pattern and check_path != str(rel_path).lstrip('./'):
                continue
            
            # Perform the pattern matching
            if fnmatch.fnmatch(check_path, pattern) or \
               fnmatch.fnmatch(check_path, f"{pattern}/*") or \
               (pattern_path.parts and fnmatch.fnmatch(check_path, str(Path('**') / pattern))):
                return True
    
    # If not ignored by patterns, check file extension (if allowed_extensions is specified)
    if allowed_extensions is not None:
        if path.is_file():
            if path.suffix.lstrip('.') not in allowed_extensions:
                return True
        elif path.is_dir():
            # Don't ignore directories when filtering by extension
            return False

    return False

def count_tokens(text):
    """Count the number of tokens in the given text."""
    enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
    return len(enc.encode(text))

def process_file(file_info):
    """Process a single file and return its content and token count."""
    file_path, relative_path, ignore_patterns, max_file_size, output_file, allowed_extensions = file_info
    
    if should_ignore(file_path, ignore_patterns, output_file, allowed_extensions):
        return None, 0
    
    if os.path.getsize(file_path) > max_file_size:
        content = f"## File: {relative_path}\n\nFile exceeds size limit. Content not included.\n\n"
        return content, count_tokens(content)
    
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
    
    return content, count_tokens(content)


def compile_project(project_path, output_file, output_format='markdown', max_file_size=1000000, allowed_extensions=None):
    """Compile project files into a single file."""
    ignore_patterns = load_cfignore(project_path)
    
    start_time = datetime.datetime.now()
    total_files = 0
    processed_files = 0
    ignored_files = 0
    total_tokens = 0
    
    file_contents = []
    
    with ThreadPoolExecutor() as executor:
        future_to_file = {}
        for root, dirs, files in os.walk(project_path):
            # Check and remove ignored directories
            dirs[:] = [d for d in dirs if not should_ignore(os.path.join(root, d), ignore_patterns, output_file, None)]
            
            for file in files:
                total_files += 1
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, project_path)
                
                if not should_ignore(file_path, ignore_patterns, output_file, allowed_extensions):
                    future = executor.submit(process_file, (file_path, relative_path, ignore_patterns, max_file_size, output_file, allowed_extensions))
                    future_to_file[future] = relative_path
                else:
                    ignored_files += 1
        
        for future in tqdm(as_completed(future_to_file), total=len(future_to_file), desc="Forging context", unit="file"):
            content, token_count = future.result()
            if content:
                file_contents.append(content)
                processed_files += 1
                total_tokens += token_count
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
        "compilation_time": compilation_time,
        "total_tokens": total_tokens,
        "allowed_extensions": list(allowed_extensions) if allowed_extensions else "All"
    }
    
    with open(output_file, 'w', encoding='utf-8') as out_file:
        if output_format == 'markdown':
            out_file.write(f"# ContextForge Compilation: {metadata['project_name']}\n\n")
            out_file.write("## Compilation Metadata\n\n")
            out_file.write(f"- Compilation Date: {metadata['compilation_date']}\n")
            out_file.write(f"- Total Files: {metadata['total_files']}\n")
            out_file.write(f"- Processed Files: {metadata['processed_files']}\n")
            out_file.write(f"- Ignored Files: {metadata['ignored_files']}\n")
            out_file.write(f"- Compilation Time: {metadata['compilation_time']:.2f} seconds\n")
            out_file.write(f"- Total Tokens: {metadata['total_tokens']}\n\n")
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
        elif output_format == 'xml':
            root = ET.Element("documents")
            for index, content in enumerate(file_contents, start=1):
                document = ET.SubElement(root, "document")
                document.set("index", str(index))
                
                # Extract file path from the content
                file_path = content.split("\n", 2)[0].replace("## File: ", "").strip()
                
                source = ET.SubElement(document, "source")
                source.text = file_path
                
                document_content = ET.SubElement(document, "document_content")
                
                # Remove the file path and location information
                content_lines = content.split("\n")[2:]  # Skip the first two lines
                cleaned_content = "\n".join(line for line in content_lines if not line.startswith("Location:"))
                
                # Remove code block markers if present
                if cleaned_content.strip().startswith("```") and cleaned_content.strip().endswith("```"):
                    cleaned_content = "\n".join(cleaned_content.split("\n")[1:-1])
                
                document_content.text = cleaned_content.strip()
            
            xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")
            out_file.write(xml_str)

    print(f"\nContextForge compilation complete:")
    print(f"- Total files: {total_files}")
    print(f"- Processed files: {processed_files}")
    print(f"- Ignored files: {ignored_files}")
    print(f"- Total tokens: {total_tokens}")
    print(f"- Compilation time: {compilation_time:.2f} seconds")
    print(f"AI-ready context saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(
        description="ContextForge: Compile project files into a single AI-ready file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
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

  Compile only Python and JavaScript files:
    %(prog)s --extensions py,js

  Compile to XML format with 2MB max file size, only including Python files:
    %(prog)s -f xml -m 2000000 --extensions py /path/to/project output.xml

  Run in watch mode:
    %(prog)s --watch    

  Get help:
    %(prog)s -h
        """
    )
    parser.add_argument("project_path", nargs='?', default=".", help="Path to the project folder (default: current directory)")
    parser.add_argument("output_file", nargs='?', default=None, help="Path to the output file (default: project_name.{format})")
    parser.add_argument("-f", "--format", choices=['markdown', 'html', 'json', 'xml'], default='markdown', help="Output format (default: %(default)s)")
    parser.add_argument("-m", "--max-file-size", type=int, default=1000000, help="Maximum file size in bytes to include in compilation (default: %(default)s)")
    parser.add_argument("--extensions", type=str, default=None, help="Comma-separated list of file extensions to include (e.g., 'py,js,md')")
    parser.add_argument("--watch", action="store_true", help="Run in watch mode, recompiling on file changes")
    args = parser.parse_args()

    # Determine the output file name
    if args.output_file is None:
        project_name = os.path.basename(os.path.abspath(args.project_path))
        args.output_file = f"{project_name}.{args.format if args.format != 'markdown' else 'md'}"
    else:
        # Ensure the file extension matches the output format
        base, ext = os.path.splitext(args.output_file)
        if args.format == 'markdown':
            args.output_file = f"{base}.md"
        elif ext.lower() != f'.{args.format}':
            args.output_file = f"{base}.{args.format}"

    # Convert extensions string to a set
    allowed_extensions = set(args.extensions.split(',')) if args.extensions else None

    if args.watch:
        print(f"Starting ContextForge in watch mode. Press Ctrl+C to stop.")
        watch_project(args.project_path, args.output_file, args.format, args.max_file_size, allowed_extensions)
    else:
        compile_project(args.project_path, args.output_file, args.format, args.max_file_size, allowed_extensions)

if __name__ == "__main__":
    main()