#!/bin/bash

# Find all files in git history bigger than 50MB
echo "Detecting large files (>50MB) in git history..."

# List files larger than 50MB (use git rev-list + git cat-file)
large_files=$(git rev-list --objects --all | \
  git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' | \
  awk '$1=="blob" && $3 > 50000000 {print $4}' | \
  sort -u)

if [ -z "$large_files" ]; then
  echo "No large files detected."
  exit 0
fi

echo "Large files to remove:"
echo "$large_files"

# Check if git filter-repo is installed
if ! command -v git-filter-repo &> /dev/null; then
  echo "Error: git-filter-repo is not installed."
  echo "Install it with: pip install git-filter-repo"
  exit 1
fi

# Build git filter-repo command arguments
filter_args=""
while IFS= read -r file; do
  if [ -n "$file" ]; then
    filter_args+=" --path \"$file\" --invert-paths"
  fi
done <<< "$large_files"

if [ -z "$filter_args" ]; then
  echo "No files to process."
  exit 0
fi

echo "Running git filter-repo with arguments:"
echo "git filter-repo$filter_args"

# Confirm before proceeding
read -p "This will rewrite git history. Continue? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
  echo "Operation cancelled."
  exit 0
fi

# Create a backup branch
echo "Creating backup branch..."
git branch backup-before-filter-repo

# Run the filter-repo command
echo "Removing large files from history..."
eval "git filter-repo$filter_args"

# Check if filter-repo succeeded
if [ $? -ne 0 ]; then
  echo "Error: git filter-repo failed."
  echo "Restoring from backup..."
  git checkout backup-before-filter-repo
  exit 1
fi

# Add remote back (filter-repo removes it)
echo "Adding remote back..."
remote_url=$(git config --get remote.origin.url 2>/dev/null || echo "")
if [ -n "$remote_url" ]; then
  git remote add origin "$remote_url"
else
  echo "Warning: Could not determine original remote URL."
  echo "You'll need to add the remote manually: git remote add origin <URL>"
fi

# Force push confirmation
echo "Repository cleaned successfully."
echo "Current branch: $(git branch --show-current)"
echo "Remote status: $(git remote -v)"

read -p "Force push to remote? This will overwrite remote history! (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
  current_branch=$(git branch --show-current)
  echo "Force pushing to origin/$current_branch..."
  git push origin "$current_branch" --force
  echo "Done cleaning and pushing."
else
  echo "Skipped force push. Run manually when ready:"
  echo "git push origin $(git branch --show-current) --force"
fi

echo "Cleanup complete!"
echo "Backup branch 'backup-before-filter-repo' created for safety."