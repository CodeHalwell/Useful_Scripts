# manipulating_files.py

This script offers small helpers for common filesystem tasks using the `pathlib` module. It demonstrates iterating
through directories, renaming files and reading or writing plain text.

## Helper Functions

- **`list_files_by_extension`** – returns all files in a directory that match a particular extension using `Path.glob`.
- **`rename_files_with_prefix`** – prepends a prefix to every file name in a directory. The function loops over
  `Path.iterdir()` and renames each path.
- **`remove_files_by_extension`** – deletes files with a given extension, showing how to call `Path.unlink`.
- **`count_lines`** – counts the number of lines in a file by iterating over it. Using a generator expression keeps
  memory usage low.
- **`read_text` / `write_text`** – simple wrappers around opening files with a context manager.

## Syntax Highlights

Each function uses type annotations to clarify argument types. The `for` loops illustrate typical iteration over
`Path` objects. Reading and writing is done with the standard `open` function, ensuring files are closed properly via
the `with` statement.

## Concepts

Understanding how to manipulate the filesystem is essential for automation scripts and data pipelines. Using
`pathlib` provides an object-oriented approach that works across operating systems. Counting lines or removing files by
pattern are everyday tasks in data preprocessing and log management.
