# 🐧🐍 Linux & Python — A Beginner's Tutorial and Reference

A sequential, example-driven guide to the commands you will actually use every day. Read top-to-bottom for a coherent tutorial, or jump to a section for a quick lookup.

---

# Part I — Linux

## 0. Mental Model First

A Linux system is essentially:

- **Files** organised in a **tree** starting at `/` (root).
- **Processes** — running programs.
- **Users** — you, running commands in a **shell** (usually `bash`).
- **Permissions** — every file/process belongs to a user, a group, and has access flags.

Everything else is just commands that manipulate these four concepts.

---

## 1. Navigating the Filesystem

### Where am I?

```bash
pwd
# /home/alice/projects
```

### What's here?

```bash
ls                  # simple listing
ls -l               # long format: permissions, size, date
ls -a               # show hidden files (starting with .)
ls -lah             # human-readable sizes, long, all
ls -lt              # sort by modification time
```

**Example output of `ls -lah`:**
```
drwxr-xr-x  2 alice alice 4.0K Oct 15 10:23 scripts
-rw-r--r--  1 alice alice 2.1K Oct 15 09:01 notes.txt
```

- `d` = directory, `-` = regular file, `l` = symlink
- `rwxr-xr-x` = owner / group / others permissions

### Changing directory

```bash
cd /etc          # absolute path
cd projects      # relative path
cd ..            # up one level
cd ../..         # up two levels
cd ~             # home directory
cd -             # previous directory
```

### Tree of a directory (if installed)

```bash
tree -L 2        # 2 levels deep
```

---

## 2. Files and Directories — Create, Copy, Move, Delete

```bash
mkdir my_project                # create a directory
mkdir -p data/raw/2024          # create nested dirs in one go

touch notes.txt                 # create empty file
cp notes.txt backup.txt         # copy a file
cp -r data/ data_backup/        # copy a directory (recursive)
mv notes.txt data/              # move or rename
mv old_name.txt new_name.txt    # rename

rm notes.txt                    # delete a file
rm -r data/                     # delete a directory recursively
rm -rf data/                    # force, no prompts (careful!)
```

> ⚠️ **Warning:** `rm -rf` is irreversible. There is no trash bin on the command line. Triple-check the path before pressing Enter.

---

## 3. Viewing Files

```bash
cat file.txt             # print whole file
less file.txt            # scrollable viewer (q to quit)
head -n 20 file.txt      # first 20 lines
tail -n 20 file.txt      # last 20 lines
tail -f logfile.log      # follow a log in real time
wc -l file.txt           # count lines
nl file.txt              # number lines
file mystery.bin         # guess file type
```

### Searching inside files

```bash
grep "error" logfile.log                # lines containing 'error'
grep -i "error" logfile.log             # case-insensitive
grep -rn "TODO" src/                    # recursive, show line numbers
grep -c "error" logfile.log             # count matches
```

### Piping and redirection

```bash
ls -l > list.txt                 # write stdout to file (overwrite)
ls -l >> list.txt                # append
grep error logfile.log > errors.txt
ps aux | grep python             # pipe output of one command to another
```

---

## 4. Finding Things

```bash
find . -name "*.py"                       # all Python files below current dir
find . -name "*.log" -mtime -7            # .log files modified in last 7 days
find . -type d -name "__pycache__"        # directories only
find . -size +100M                        # files larger than 100 MB

which python                              # where is this executable?
whereis python
locate notes.txt                          # uses a pre-built index (fast)
```

---

## 5. Permissions and Ownership

```bash
chmod +x script.sh              # make executable
chmod 755 script.sh             # rwxr-xr-x (owner rwx, group rx, others rx)
chmod 644 data.txt              # rw-r--r--
chown alice:alice file.txt      # change owner:group (needs sudo)
```

Numeric permissions: `r=4, w=2, x=1`. So `7 = rwx`, `6 = rw-`, `5 = r-x`, `4 = r--`.

---

## 6. Processes

```bash
ps aux                  # all running processes
top                     # live process viewer (q to quit)
htop                    # nicer top (if installed)
kill 12345              # send SIGTERM to PID 12345
kill -9 12345           # force kill
pkill -f "python app"   # kill by name/pattern
```

### Background jobs

```bash
python train.py &       # run in background
jobs                    # list jobs in the current shell
fg                      # bring last job to foreground
Ctrl+C                  # interrupt foreground process
Ctrl+Z                  # suspend
bg                      # resume last suspended job in background
```

---

## 7. Archives and Compression

```bash
tar -czvf archive.tar.gz mydir/     # create a .tar.gz
tar -xzvf archive.tar.gz            # extract
zip -r archive.zip mydir/           # zip
unzip archive.zip                   # unzip
```

Flags: `c`=create, `x`=extract, `z`=gzip, `v`=verbose, `f`=file.

---

## 8. Networking Basics

```bash
ping google.com
curl https://example.com                 # fetch a URL
curl -O https://example.com/file.zip     # download with same name
wget https://example.com/file.zip        # similar to curl -O
ssh user@server.com                      # connect to a remote machine
scp file.txt user@server:/path/          # copy file to remote
rsync -av src/ user@server:/dest/        # sync (fast, resumable)
```

---

## 9. Package Management (Debian/Ubuntu)

```bash
sudo apt update            # refresh package lists
sudo apt upgrade           # upgrade installed packages
sudo apt install git       # install
sudo apt remove git        # uninstall
apt search numpy           # search
```

On Fedora: `dnf`. On Arch: `pacman`. On macOS you'd use `brew` (if you're curious).

---

## 10. Text Editors

### Nano — easiest for beginners

```bash
nano file.txt
# Ctrl+O — save, Ctrl+X — exit
```

### Vim — powerful, steep learning curve

```bash
vim file.txt
# i — insert mode
# Esc — normal mode
# :w — save, :q — quit, :wq — save and quit, :q! — quit without saving
```

### VS Code — GUI editor with built-in terminal

```bash
code .
```

---

## 11. Handy One-Liners

```bash
history                          # your command history
!!                               # repeat last command
!grep                            # repeat last command starting with 'grep'
Ctrl+R                           # search history interactively
alias ll='ls -lah'               # define a shortcut
echo $PATH                       # show the PATH variable
export MYVAR="hello"             # set a variable
env                              # list environment variables
df -h                            # disk free space
du -sh dir/                      # directory size
free -h                          # RAM usage
uname -a                         # system info
```

---

## 12. The "Oh No" Cheat Sheet

| Situation | Command |
|---|---|
| A process is stuck | `Ctrl+C` |
| Need to cancel a running command in nano | `Ctrl+X` → `N` |
| Accidentally opened vim, can't escape | `Esc` `:q!` `Enter` |
| Terminal output flooded | `Ctrl+L` (clear) or `clear` |
| Don't know how to use a command | `man command` or `command --help` |

