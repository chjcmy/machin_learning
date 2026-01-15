
import os
import ast
import json
import sys
import pkgutil

def get_stdlib_names():
    if sys.version_info >= (3, 10):
        return sys.stdlib_module_names
    else:
        return {
            'abc', 'aifc', 'argparse', 'array', 'ast', 'asynchat', 'asyncio', 'asyncore', 'atexit', 'audioop',
            'base64', 'bdb', 'binascii', 'binhex', 'bisect', 'builtins', 'bz2', 'cProfile', 'calendar', 'cgi',
            'cgitb', 'chunk', 'cmath', 'cmd', 'code', 'codecs', 'codeop', 'collections', 'colorsys', 'compileall',
            'concurrent', 'configparser', 'contextlib', 'contextvars', 'copy', 'copyreg', 'crypt', 'csv', 'ctypes',
            'curses', 'dataclasses', 'datetime', 'dbm', 'decimal', 'difflib', 'dis', 'distutils', 'doctest',
            'dummy_threading', 'email', 'encodings', 'ensurepip', 'enum', 'errno', 'faulthandler', 'fcntl',
            'filecmp', 'fileinput', 'fnmatch', 'formatter', 'fractions', 'ftplib', 'functools', 'gc', 'getopt',
            'getpass', 'gettext', 'glob', 'graphlib', 'grp', 'gzip', 'hashlib', 'heapq', 'hmac', 'html', 'http',
            'imaplib', 'imghdr', 'imp', 'importlib', 'inspect', 'io', 'ipaddress', 'itertools', 'json', 'keyword',
            'lib2to3', 'linecache', 'locale', 'logging', 'lzma', 'mailbox', 'mailcap', 'marshal', 'math',
            'mimetypes', 'mmap', 'modulefinder', 'msilib', 'msvcrt', 'multiprocessing', 'netrc', 'nntplib',
            'ntpath', 'numbers', 'operator', 'optparse', 'os', 'ossaudiodev', 'parser', 'pathlib', 'pdb', 'pickle',
            'pickletools', 'pipes', 'pkgutil', 'platform', 'plistlib', 'poplib', 'posix', 'posixpath', 'pprint',
            'profile', 'pstats', 'pty', 'pwd', 'py_compile', 'pyclbr', 'pydoc', 'queue', 'quopri', 'random', 're',
            'readline', 'reprlib', 'resource', 'rlcompleter', 'runpy', 'sched', 'secrets', 'select', 'selectors',
            'shelve', 'shlex', 'shutil', 'signal', 'site', 'smtpd', 'smtplib', 'sndhdr', 'socket', 'socketserver',
            'spwd', 'sqlite3', 'ssl', 'stat', 'statistics', 'string', 'stringprep', 'struct', 'subprocess', 'sunau',
            'symbol', 'symtable', 'sys', 'sysconfig', 'syslog', 'tabnanny', 'tarfile', 'telnetlib', 'tempfile',
            'termios', 'test', 'textwrap', 'threading', 'time', 'timeit', 'tkinter', 'token', 'tokenize', 'trace',
            'traceback', 'tracemalloc', 'tty', 'turtle', 'turtledemo', 'types', 'typing', 'unicodedata', 'unittest',
            'urllib', 'uu', 'uuid', 'venv', 'warnings', 'wave', 'weakref', 'webbrowser', 'winsound', 'wsgiref',
            'xdrlib', 'xml', 'xmlrpc', 'zipapp', 'zipfile', 'zipimport', 'zlib', 'zoneinfo'
        }

STDLIB = get_stdlib_names()

def extract_imports_from_code(code):
    imports = set()
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module.split('.')[0])
    except SyntaxError:
        pass # Ignore syntax errors in snippets
    return imports

def scan_file(filepath):
    imports = set()
    _, ext = os.path.splitext(filepath)
    
    if ext == '.py':
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                code = f.read()
                imports.update(extract_imports_from_code(code))
        except Exception:
            pass # Ignore files that can't be read
            
    elif ext == '.ipynb':
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                notebook = json.load(f)
                for cell in notebook.get('cells', []):
                    if cell.get('cell_type') == 'code':
                        source = "".join(cell.get('source', []))
                        imports.update(extract_imports_from_code(source))
        except Exception:
            pass

    return imports

def main():
    root_dir = r"c:\Users\UserK\machin_learning"
    all_imports = set()
    
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if '.venv' in dirpath or '.git' in dirpath:
            continue
            
        for filename in filenames:
            if filename.endswith('.py') or filename.endswith('.ipynb'):
                filepath = os.path.join(dirpath, filename)
                all_imports.update(scan_file(filepath))
    
    # Filter stdlib and local imports (heuristic)
    third_party = {imp for imp in all_imports if imp not in STDLIB and imp}
    
    # Ad-hoc mapping for known packages (sklearn -> scikit-learn)
    mappings = {
        'sklearn': 'scikit-learn',
        'PIL': 'pillow',
        'cv2': 'opencv-python',
        'skimage': 'scikit-image',
        'yaml': 'pyyaml',
        'bs4': 'beautifulsoup4'
    }
    
    final_requirements = set()
    for imp in third_party:
        # Ignore local modules (simple check: if it matches a filename in root)
        # For now, just add mapped name or original
        req_name = mappings.get(imp, imp)
        # Basic cleanup: remove obviously local names if needed, but for now specific filtering:
        # Filter out common false positives or local scripts if they are in the list?
        # Assuming most are libs.
        final_requirements.add(req_name)

    # Output for the next step
    print("\n".join(sorted(final_requirements)))

if __name__ == "__main__":
    main()
