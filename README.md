<img src="https://github.com/seeschweiler/contextforge/blob/main/cflogo.png" alt="ContextForge Logo" width="350">

# ContextForge

ContextForge is a powerful and flexible command-line tool designed to compile the contents of a development project into a single, well-structured file. This compiled output is ideal for use as input to large language models (LLMs) like GPT, making it easier to provide comprehensive project context in a single prompt.

As LLMs continue to evolve, we're seeing a significant increase in their context window sizes. This expansion allows these models to process and understand larger amounts of information at once, opening up new possibilities for developers and AI practitioners. ContextForge is at the forefront of this revolution, enabling users to leverage these expanded context windows to their full potential.

With ContextForge, you can now compile your entire project—including code, documentation, and configuration files—into a single, coherent document. This comprehensive compilation allows you to provide LLMs with a complete picture of your project, leading to more accurate and contextually relevant responses. Whether you're seeking code suggestions, architectural advice, or deep project analysis, ContextForge ensures that the LLM has access to all the necessary information.

Key benefits of using ContextForge with large context window LLMs include:

1. Holistic Understanding: LLMs can grasp the full scope of your project, including intricate relationships between different components.
2. Improved Accuracy: With access to more context, LLMs can provide more precise and project-specific suggestions and analyses.
3. Time Efficiency: Instead of manually selecting and pasting relevant parts of your project, ContextForge automates the process of creating a comprehensive context.
4. Consistency: Ensure that every interaction with the LLM is based on the same, complete project context, leading to more consistent and coherent assistance.
5. Scalability: As your project grows, ContextForge scales with it, always providing the most up-to-date and complete context to the LLM.

By bridging the gap between expansive codebases and the growing capabilities of LLMs, ContextForge empowers developers to harness the full potential of AI assistance in their development workflows. Whether you're working on a small script or a large-scale application, ContextForge is an essential tool for maximizing the benefits of large context window LLMs in your development process.

## Table of Contents

- [ContextForge](#contextforge)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Configuration](#configuration)
    - [.cfignore File](#cfignore-file)
  - [Output Formats](#output-formats)
  - [Examples](#examples)
  - [Contributing](#contributing)

## Features

- **Project Compilation**: Recursively scans and compiles the contents of a project directory into a single file.
- **Multiple Output Formats**: Supports Markdown, HTML, JSON, and XML output formats.
- **Syntax Highlighting**: Automatically detects and applies appropriate language syntax highlighting for code files.
- **Ignore Patterns**: Supports `.cfignore` files to exclude specific files or directories from compilation, similar to `.gitignore`.
- **File Size Limit**: Option to set a maximum file size for inclusion in the compilation.
- **Metadata Inclusion**: Adds useful metadata about the compilation process to the output.
- **Parallel Processing**: Uses multi-threading to speed up the compilation process for large projects.
- **Progress Tracking**: Displays a progress bar during compilation.
- **Smart File Naming**: Automatically uses the project folder name as the default output file name.
- **Consistent File Extensions**: Ensures the output file extension matches the chosen format.

## Installation

1. Ensure you have Python 3.6 or later installed on your system.

2. Clone the ContextForge repository:
   ```
   git clone https://github.com/yourusername/contextforge.git
   cd contextforge
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

The basic usage of ContextForge is as follows:

```
python contextforge.py [project_path] [output_file] [-f FORMAT] [-m MAX_FILE_SIZE]
```

- `project_path`: Path to the project folder (default: current directory)
- `output_file`: Path to the output file (default: project_name.{format})
- `-f, --format`: Output format (markdown, html, json, or xml; default: markdown)
- `-m, --max-file-size`: Maximum file size in bytes to include (default: 1000000)

For more information and options, use the help command:

```
python contextforge.py -h
```

## Configuration

### .cfignore File

ContextForge supports a `.cfignore` file in the root of your project directory. This file works similarly to `.gitignore`, allowing you to specify patterns for files and directories that should be excluded from the compilation.

Example `.cfignore` file:

```
# Ignore all .log files
*.log

# Ignore the entire 'node_modules' directory
node_modules/

# Ignore a specific file
secrets.txt

# Ignore all files in a specific directory
build/*
```

## Output Formats

ContextForge supports four output formats:

1. **Markdown** (default): A well-structured Markdown file with appropriate code blocks and syntax highlighting.
2. **HTML**: An HTML file with syntax-highlighted code blocks, suitable for viewing in a web browser.
3. **JSON**: A JSON file containing the project structure and file contents, useful for programmatic processing.
4. **XML**: An XML file with a structured representation of the project, ideal for parsing and processing with XML tools.

## Examples

1. Compile the current directory to the default output file (project_name.md):
   ```
   python contextforge.py
   ```

2. Compile a specific project to a custom output file:
   ```
   python contextforge.py /path/to/project custom_output.md
   ```

3. Compile to HTML format:
   ```
   python contextforge.py -f html
   ```

4. Compile with a 500KB max file size:
   ```
   python contextforge.py -m 500000
   ```

5. Compile to JSON format with a 2MB max file size:
   ```
   python contextforge.py -f json -m 2000000 /path/to/project
   ```

6. Compile to XML format:
   ```
   python contextforge.py -f xml /path/to/project
   ```

## Contributing

Contributions to ContextForge are welcome! Please feel free to submit a Pull Request.